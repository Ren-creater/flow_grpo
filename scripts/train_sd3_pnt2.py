from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import gc
import psutil
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../TPDM/src")))
from models.stable_diffusion_3.modeling_sd3_pnt import init_time_predictor
from models.reference_distributions import get_ref_beta
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pnt_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_pnt_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def cleanup_memory():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    # Set up scheduler state for sde_step_with_logprob
    current_batch_size = sample["latents"].shape[0]
    current_timesteps = sample["timesteps"][:, j]  # timesteps for step j across current batch
    current_sigmas = sample["sigmas"][:, j]        # sigmas for step j across current batch
    next_sigmas = sample["sigmas"][:, j + 1]       # sigmas for step j+1 across current batch
    
    # Ensure all tensors are on the correct device
    device = sample["latents"].device
    current_timesteps = current_timesteps.to(device)
    current_sigmas = current_sigmas.to(device)
    next_sigmas = next_sigmas.to(device)
    
    # Create the scheduler state that sde_step_with_logprob expects
    pipeline.scheduler.index_for_timestep = [{} for _ in range(current_batch_size)]
    
    # Set up the mapping for the current timestep
    for batch_idx in range(current_batch_size):
        timestep_val = current_timesteps[batch_idx].item()
        pipeline.scheduler.index_for_timestep[batch_idx][timestep_val] = j
    
    # Set up sigmas list where sigmas[step][batch_idx] gives the sigma value
    # We need at least steps j and j+1, plus sigma[1] for sigma_max
    max_step = max(j + 1, 1)
    pipeline.scheduler.sigmas = [torch.zeros(current_batch_size, device=device) for _ in range(max_step + 1)]
    pipeline.scheduler.sigmas[j] = current_sigmas
    pipeline.scheduler.sigmas[j + 1] = next_sigmas
    
    # Extract sigma_max from the sample's sigmas - use step 1 sigmas as sigma_max
    if sample["sigmas"].shape[1] > 1:
        sigma_max = sample["sigmas"][:, 1].to(device)  # Use second timestep sigmas as sigma_max
    else:
        sigma_max = torch.ones_like(current_sigmas)
    pipeline.scheduler.sigmas[1] = sigma_max
    
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def compute_time_predictor_log_prob(pipeline, sample, j, embeds, pooled_embeds, config):
    """
    Compute the log probability of the time predictor for step j.
    This function uses the saved hidden states and temporal embeddings to recompute
    the time predictor logprob for a specific sigma transition.
    """
    current_batch_size = sample["latents"].shape[0]
    device = sample["latents"].device
    
    # Get current and next sigmas for this timestep
    current_sigmas = sample["sigmas"][:, j]        # sigmas for step j across current batch
    next_sigmas = sample["sigmas"][:, j + 1]       # sigmas for step j+1 across current batch
    
    # Use stored hidden states and temporal embeddings from the sampling phase
    # Move to device only once and reuse
    hidden_states_combined = sample["hidden_states_combineds"][:, j]
    temb = sample["tembs"][:, j]
    
    # Move to GPU only if not already there, and convert dtype efficiently
    if hidden_states_combined.device != device:
        hidden_states_combined = hidden_states_combined.to(device, dtype=torch.float32, non_blocking=True)
    else:
        hidden_states_combined = hidden_states_combined.to(dtype=torch.float32)
        
    if temb.device != device:
        temb = temb.to(device, dtype=torch.float32, non_blocking=True)
    else:
        temb = temb.to(dtype=torch.float32)
    
    # Call the time predictor to get alpha and beta
    time_preds = pipeline.time_predictor(hidden_states_combined, temb)
    
    # Build list of log probabilities and construct final tensor from gradient-enabled tensors
    log_probs_list = []
    
    for i, (param1, param2) in enumerate(time_preds):
        # Skip log prob computation if sigma is below threshold (following modeling_sd3_pnt.py pattern)
        if current_sigmas[i] < pipeline.min_sigma:
            # Use a zero tensor that maintains gradients from time_preds
            zero_logprob = param1 * 0.0  # This maintains gradients from the time predictor
            log_probs_list.append(zero_logprob)
            continue
            
        if pipeline.prediction_type == "alpha_beta":
            alpha, beta = param1, param2
        elif pipeline.prediction_type == "mode_concentration":
            alpha = param1 * (param2 - 2) + 1
            beta = (1 - param1) * (param2 - 2) + 1
        
        # Validate alpha and beta parameters before creating Beta distribution
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)
        
        # Check for any invalid values in parameters
        if torch.isnan(alpha) or torch.isinf(alpha) or torch.isnan(beta) or torch.isinf(beta):
            # Use a zero tensor that maintains gradients from time_preds
            zero_logprob = param1 * 0.0  # This maintains gradients from the time predictor
            log_probs_list.append(zero_logprob)
            continue
        
        # Validate sigma values before ratio calculation
        if torch.isnan(current_sigmas[i]) or torch.isinf(current_sigmas[i]) or \
           torch.isnan(next_sigmas[i]) or torch.isinf(next_sigmas[i]) or \
           current_sigmas[i] <= 0:
            # Use a zero tensor that maintains gradients from time_preds
            zero_logprob = param1 * 0.0  # This maintains gradients from the time predictor
            log_probs_list.append(zero_logprob)
            continue
        
        beta_dist = torch.distributions.Beta(alpha, beta)
        
        # Calculate the ratio from the stored sigmas
        if pipeline.relative:
            ratio = next_sigmas[i] / current_sigmas[i]
        else:
            ratio = current_sigmas[i] - next_sigmas[i]
        
        # Clamp ratio and check for NaN/inf
        ratio = torch.clamp(ratio, min=pipeline.epsilon, max=1 - pipeline.epsilon)
        if torch.isnan(ratio) or torch.isinf(ratio):
            # Use a zero tensor that maintains gradients from time_preds
            zero_logprob = param1 * 0.0  # This maintains gradients from the time predictor
            log_probs_list.append(zero_logprob)
            continue
        
        # Compute the log probability
        time_predictor_log_prob = beta_dist.log_prob(ratio)
        log_probs_list.append(time_predictor_log_prob)
    
    # Stack all log probabilities into a single tensor that maintains gradients
    time_predictor_log_probs = torch.stack(log_probs_list, dim=0)
    
    return time_predictor_log_probs

def compute_time_predictor_kl_divergence(pipeline, sample, j, embeds, pooled_embeds, config):
    """
    Compute the KL divergence between the time predictor's predicted Beta distribution
    and a reference Beta distribution for step j.
    """
    current_batch_size = sample["latents"].shape[0]
    device = sample["latents"].device
    
    # Get current sigmas for this timestep to compute reference distribution
    current_sigmas = sample["sigmas"][:, j]        # sigmas for step j across current batch
    
    # Use stored hidden states and temporal embeddings from the sampling phase
    # Optimize tensor transfers similar to compute_time_predictor_log_prob
    hidden_states_combined = sample["hidden_states_combineds"][:, j]
    temb = sample["tembs"][:, j]
    
    # Move to GPU only if not already there, and convert dtype efficiently
    if hidden_states_combined.device != device:
        hidden_states_combined = hidden_states_combined.to(device, dtype=torch.float32, non_blocking=True)
    else:
        hidden_states_combined = hidden_states_combined.to(dtype=torch.float32)
        
    if temb.device != device:
        temb = temb.to(device, dtype=torch.float32, non_blocking=True)
    else:
        temb = temb.to(dtype=torch.float32)
    
    # Call the time predictor to get alpha and beta
    time_preds = pipeline.time_predictor(hidden_states_combined, temb)
    
    # Build list of KL divergences and construct final tensor from gradient-enabled tensors
    kl_divs_list = []
    
    for i, (param1, param2) in enumerate(time_preds):
        # Skip KL computation if sigma is below threshold (following modeling_sd3_pnt.py pattern)
        if current_sigmas[i] < pipeline.min_sigma:
            # Use a zero tensor that maintains gradients from time_preds
            zero_kl = param1 * 0.0  # This maintains gradients from the time predictor
            kl_divs_list.append(zero_kl)
            continue
            
        if pipeline.prediction_type == "alpha_beta":
            alpha, beta = param1, param2
        elif pipeline.prediction_type == "mode_concentration":
            alpha = param1 * (param2 - 2) + 1
            beta = (1 - param1) * (param2 - 2) + 1
        
        # Validate sigma values before using them for reference distribution
        if torch.isnan(current_sigmas[i]) or torch.isinf(current_sigmas[i]) or current_sigmas[i] <= 0:
            # Use a zero tensor that maintains gradients from time_preds
            zero_kl = param1 * 0.0  # This maintains gradients from the time predictor
            kl_divs_list.append(zero_kl)
            continue
        
        # Get reference distribution parameters using the same logic as in modeling_sd3_pnt.py
        if pipeline.relative:
            # Use the get_ref_beta function to get reference alpha/beta based on current sigma
            # Reshape sigma for get_ref_beta function (expects 1D tensor)
            sigma_input = current_sigmas[i:i+1]  # Shape: (1,)
            ref_alpha, ref_beta = get_ref_beta(sigma_input)
            ref_alpha, ref_beta = ref_alpha[0], ref_beta[0]  # Extract scalar values
        else:
            # Use fixed reference distribution for non-relative case
            ref_alpha, ref_beta = 1.4, 11.2
        
        # Validate and clamp all parameters to ensure they're valid for Beta distribution
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)
        ref_alpha = torch.clamp(ref_alpha, min=1e-6)
        ref_beta = torch.clamp(ref_beta, min=1e-6)
        
        # Check for any invalid values (NaN, inf)
        if torch.isnan(alpha) or torch.isinf(alpha) or torch.isnan(beta) or torch.isinf(beta) or \
           torch.isnan(ref_alpha) or torch.isinf(ref_alpha) or torch.isnan(ref_beta) or torch.isinf(ref_beta):
            # Use a zero tensor that maintains gradients from time_preds
            zero_kl = param1 * 0.0  # This maintains gradients from the time predictor
            kl_divs_list.append(zero_kl)
        else:
            # Create distributions and compute KL divergence
            predicted_dist = torch.distributions.Beta(alpha, beta)
            ref_dist = torch.distributions.Beta(ref_alpha, ref_beta)
            kl_div = torch.distributions.kl_divergence(predicted_dist, ref_dist)
            # Final check for NaN/inf in the result
            if torch.isnan(kl_div) or torch.isinf(kl_div):
                # Use a zero tensor that maintains gradients from time_preds
                zero_kl = param1 * 0.0  # This maintains gradients from the time predictor
                kl_divs_list.append(zero_kl)
            else:
                kl_divs_list.append(kl_div)
    
    # Stack all KL divergences into a single tensor that maintains gradients
    kl_divergences = torch.stack(kl_divs_list, dim=0)
    
    return kl_divergences

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, get_trainable_params_fn):
    if config.train.ema:
        current_trainable_params = get_trainable_params_fn()
        ema.copy_ema_to(current_trainable_params, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(current_trainable_params)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, pipeline, global_step, accelerator, ema, get_trainable_params_fn, config, is_time_predictor_only_phase=False):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            current_trainable_params = get_trainable_params_fn()
            ema.copy_ema_to(current_trainable_params, store_temp=True)
        
        # Only save transformer if it's being trained (not in time_predictor-only phase)
        if not is_time_predictor_only_phase:
            unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
            logger.info(f"Saved transformer weights to {save_root_lora}")
        else:
            logger.info(f"Skipping transformer save during time_predictor-only training phase")
        
        # Always save time_predictor weights (it's being trained in both phases)
        time_predictor_path = os.path.join(save_root, "time_predictor.pt")
        torch.save(unwrap_model(pipeline.time_predictor, accelerator).state_dict(), time_predictor_path)
        logger.info(f"Saved time_predictor weights to {time_predictor_path}")
        
        if config.train.ema:
            ema.copy_temp_to(current_trainable_params)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # Check if we need special DDP handling for time_predictor-only training
    time_predictor_only_epochs = getattr(config.train, 'time_predictor_only_epochs', 0)
    
    # Prepare kwargs for Accelerator initialization
    accelerator_kwargs = {
        "mixed_precision": config.mixed_precision,
        "project_config": accelerator_config,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps * num_train_timesteps,
    }
    
    # Add DDP kwargs if we have time_predictor-only training
    if time_predictor_only_epochs > 0:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]

    accelerator = Accelerator(**accelerator_kwargs)
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )    

    init_time_predictor(pipeline, config.sd3_checkpoint_path, use_vit_predictor=config.use_vit_predictor)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.time_predictor.requires_grad_(True)

    # Freeze VAE and text encoders if needed, and set up LoRA as before
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    # model.safety_checker = None  # If applicable
    # Set up any additional model configs as needed
    
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    # Move transformer to device but keep in original precision
    # Mixed precision will be handled by autocast during forward passes
    pipeline.transformer.to(accelerator.device)
    # Keep time_predictor in float32 for training stability
    pipeline.time_predictor.to(accelerator.device, dtype=torch.float32)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    time_predictor_parameters = list(pipeline.time_predictor.parameters())
    all_trainable_parameters = transformer_trainable_parameters + time_predictor_parameters
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(all_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        all_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Helper functions for time_predictor-only training
    # 
    # TIME_PREDICTOR-ONLY TRAINING FEATURE:
    # This feature allows training only the time_predictor for the first few epochs while keeping
    # the rest of the model (transformer) frozen. This can be useful for:
    # 1. Warm-up phase: Let time_predictor learn basic time dynamics before joint training
    # 2. Faster experimentation: Test time_predictor changes without expensive transformer training
    # 3. Stability: Ensure time_predictor is reasonably trained before full model training
    #
    # Configuration:
    # - Set config.train.time_predictor_only_epochs = N (where N > 0) to enable this feature
    # - Set config.train.time_predictor_only_epochs = 0 to disable (default)
    #
    # TIME_PREDICTOR KL REGULARIZATION:
    # KL regularization encourages the time predictor's Beta distributions to stay close to
    # a reference distribution based on the original scheduler dynamics. This helps:
    # 1. Prevent the time predictor from making extreme predictions
    # 2. Maintain reasonable timestep transitions
    # 3. Stabilize training by providing a prior over time dynamics
    #
    # Configuration:
    # - Set config.train.time_predictor_kl_weight = 0.01 (or desired value) to control strength
    # - Set config.train.time_predictor_kl_weight = 0.0 to disable KL regularization
    #
    # During time_predictor-only phase:
    # - Only time_predictor parameters have requires_grad=True
    # - Optimizer contains only time_predictor parameters
    # - Checkpoints save only time_predictor weights (transformer is skipped)
    # - WandB logging includes "time_predictor_only_phase" flag
    # - Both GRPO loss and KL regularization are applied to time_predictor
    #
    # At transition (epoch == time_predictor_only_epochs):
    # - Saves final time_predictor-only checkpoint
    # - Unfreezes transformer parameters (LoRA or full based on config.use_lora)
    # - Creates new optimizer with all trainable parameters
    # - Note: Optimizer state is lost during transition
    #
    def get_current_trainable_parameters():
        """Get current trainable parameters based on the training phase"""
        current_transformer_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        return current_transformer_params + time_predictor_parameters
    
    def freeze_transformer():
        """Freeze transformer parameters for time_predictor-only training"""
        for param in transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_transformer():
        """Unfreeze transformer parameters after time_predictor-only training"""
        if config.use_lora:
            # For LoRA, only unfreeze LoRA parameters
            for param in transformer.parameters():
                if hasattr(param, 'is_lora') and param.is_lora:
                    param.requires_grad = True
        else:
            # For full fine-tuning, unfreeze all transformer parameters
            for param in transformer.parameters():
                param.requires_grad = True
    
    def create_time_predictor_only_optimizer():
        """Create optimizer for time_predictor-only training"""
        return optimizer_cls(
            time_predictor_parameters,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
    
    def create_full_optimizer():
        """Create optimizer for full training (transformer + time_predictor)"""
        current_trainable = list(filter(lambda p: p.requires_grad, transformer.parameters())) + time_predictor_parameters
        return optimizer_cls(
            current_trainable,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, time_predictor, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, pipeline.time_predictor, optimizer, train_dataloader, test_dataloader)
    
    # Reassign the prepared time_predictor back to pipeline
    pipeline.time_predictor = time_predictor

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        # Handle time_predictor-only training phase
        time_predictor_only_epochs = getattr(config.train, 'time_predictor_only_epochs', 0)
        is_time_predictor_only_phase = epoch < time_predictor_only_epochs
        
        # Switch from time_predictor-only to full training if needed
        if epoch == time_predictor_only_epochs and time_predictor_only_epochs > 0:
            logger.info(f"Switching from time_predictor-only to full training at epoch {epoch}")
            
            # Save checkpoint before switching (time_predictor-only final state)
            if accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, pipeline, global_step, accelerator, ema, get_current_trainable_parameters, config, is_time_predictor_only_phase=True)
                logger.info("Saved time_predictor-only checkpoint before switching to full training")
            
            # Unfreeze transformer parameters
            unfreeze_transformer()
            # Create new optimizer with all trainable parameters
            new_optimizer = create_full_optimizer()
            # Replace the old optimizer (note: this will lose optimizer state)
            optimizer = new_optimizer
            # Re-prepare the optimizer with accelerator
            optimizer = accelerator.prepare(optimizer)
            
            # Re-initialize EMA with new parameter set
            new_trainable_params = get_current_trainable_parameters()
            ema = EMAModuleWrapper(new_trainable_params, decay=0.9, update_step_interval=8, device=accelerator.device)
            logger.info("Successfully switched to full training mode and re-initialized EMA")
        
        # For epoch 0 in time_predictor_only mode, freeze transformer
        if epoch == 0 and time_predictor_only_epochs > 0:
            logger.info(f"Starting time_predictor-only training for {time_predictor_only_epochs} epochs")
            freeze_transformer()
            # Create time_predictor-only optimizer
            time_predictor_optimizer = create_time_predictor_only_optimizer()
            # Replace the optimizer
            optimizer = time_predictor_optimizer
            # Re-prepare the optimizer with accelerator
            optimizer = accelerator.prepare(optimizer)
            
            # Re-initialize EMA with only time_predictor parameters
            ema = EMAModuleWrapper(time_predictor_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
            logger.info("Successfully switched to time_predictor-only training mode and re-initialized EMA")

        #################### EVAL ####################
        pipeline.transformer.eval()
        pipeline.time_predictor.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, get_current_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, pipeline, global_step, accelerator, ema, get_current_trainable_parameters, config, is_time_predictor_only_phase)

        #################### SAMPLING ####################-
        pipeline.transformer.eval()
        pipeline.time_predictor.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs, time_predictor_log_probs, timesteps_per_sample, all_sigmas_per_step, hidden_states_combineds, tembs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            time_predictor_log_probs = torch.stack(time_predictor_log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            # Stack sigmas to match timesteps and latents structure
            # all_sigmas_per_step contains sigma values for each step, we need to stack them
            sigmas = torch.stack(all_sigmas_per_step, dim=1)  # (batch_size, num_steps + 1)

            # Stack timesteps - this should work if scheduler.timesteps has consistent shapes
            timesteps = torch.stack(pipeline.scheduler.timesteps, dim=1)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "timesteps_per_sample": timesteps_per_sample,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "time_predictor_log_probs": time_predictor_log_probs,
                    "hidden_states_combineds": hidden_states_combineds,
                    "tembs": tembs,
                    "rewards": rewards,
                    "sigmas": sigmas,  # sigma values for each timestep (needs num_steps + 1 for next_sigma access)
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            # Keep rewards for WandB logging - will be cleaned up later
        
        # Force garbage collection after reward computation
        gc.collect()

        # Pad tensors to the same length before collation to handle variable timesteps
        # Note: sigmas has shape (batch_size, num_steps + 1) while others have (batch_size, num_steps)
        max_timesteps = max(s["latents"].shape[1] for s in samples)  # This is num_steps
        max_timesteps_sigmas = max_timesteps + 1  # For sigmas which need num_steps + 1
        
        for sample in samples:
            current_timesteps = sample["latents"].shape[1]
            if current_timesteps < max_timesteps:
                pad_size = max_timesteps - current_timesteps
                
                # Use more memory-efficient padding by pre-allocating full-size tensors
                # and copying data instead of concatenating
                for key in ["latents", "next_latents"]:
                    if key in sample:
                        original_tensor = sample[key]
                        actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                        # Create new tensor with full size
                        full_shape = [original_tensor.shape[0], max_timesteps] + list(original_tensor.shape[2:])
                        new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                        # Copy original data using actual tensor dimensions
                        new_tensor[:, :actual_timesteps] = original_tensor
                        sample[key] = new_tensor
                        del original_tensor  # Explicit cleanup
                
                # Same for log_probs and time_predictor_log_probs
                for logprob_key in ["log_probs", "time_predictor_log_probs"]:
                    if logprob_key in sample:
                        original_tensor = sample[logprob_key]
                        actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                        full_shape = [original_tensor.shape[0], max_timesteps]
                        new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                        new_tensor[:, :actual_timesteps] = original_tensor
                        sample[logprob_key] = new_tensor
                        del original_tensor  # Explicit cleanup
                
                # More memory-efficient padding for hidden_states_combineds and tembs 
                for tensor_key in ["hidden_states_combineds", "tembs"]:
                    if tensor_key in sample:
                        original_tensor = sample[tensor_key]
                        actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                        full_shape = [original_tensor.shape[0], max_timesteps] + list(original_tensor.shape[2:])
                        new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                        new_tensor[:, :actual_timesteps] = original_tensor
                        sample[tensor_key] = new_tensor
                        del original_tensor  # Explicit cleanup
                
                # Pad timesteps - use the last timestep value for padding
                if "timesteps" in sample:
                    original_tensor = sample["timesteps"]
                    actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                    new_tensor = torch.zeros([original_tensor.shape[0], max_timesteps], 
                                           device=original_tensor.device, dtype=original_tensor.dtype)
                    new_tensor[:, :actual_timesteps] = original_tensor
                    # Fill padding with last timestep value
                    if actual_timesteps > 0:
                        last_timestep = original_tensor[:, -1:]
                        pad_size_actual = max_timesteps - actual_timesteps
                        new_tensor[:, actual_timesteps:] = last_timestep.repeat(1, pad_size_actual)
                    sample["timesteps"] = new_tensor
                    del original_tensor  # Explicit cleanup
                
                # Pad sigmas - special case since it has num_steps + 1 elements
                if "sigmas" in sample:
                    original_tensor = sample["sigmas"]
                    actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                    new_tensor = torch.zeros([original_tensor.shape[0], max_timesteps_sigmas], 
                                           device=original_tensor.device, dtype=original_tensor.dtype)
                    new_tensor[:, :actual_timesteps] = original_tensor
                    # Fill padding with last sigma value
                    if actual_timesteps > 0:
                        last_sigma = original_tensor[:, -1:]
                        pad_size_actual = max_timesteps_sigmas - actual_timesteps
                        new_tensor[:, actual_timesteps:] = last_sigma.repeat(1, pad_size_actual)
                    sample["sigmas"] = new_tensor
                    del original_tensor  # Explicit cleanup

        # Force garbage collection before collation
        gc.collect()

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        
        # Clean up rewards and images after WandB logging
        del rewards, reward_metadata
        del images  # Free large image tensor
        gc.collect()
        
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Get actual number of timesteps from the samples data
        actual_num_timesteps = samples["latents"].shape[1]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, actual_num_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    "time_predictor_only_phase": is_time_predictor_only_phase,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        
        # Additional cleanup to prevent memory leaks
        gc.collect()

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        # Handle device mismatch: some tensors are on CPU (hidden_states_combineds, tembs) due to memory optimization
        # Convert mask to CPU once to avoid repeated transfers
        cpu_mask = mask.cpu()
        filtered_samples = {}
        for k, v in samples.items():
            if k in ["hidden_states_combineds", "tembs"]:
                # These tensors are on CPU, so use CPU mask
                filtered_samples[k] = v[cpu_mask]
            else:
                # Other tensors are on GPU, use GPU mask
                filtered_samples[k] = v[mask]
        samples = filtered_samples

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        #assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            cpu_perm = perm.cpu()  # Create CPU version for CPU tensors
            
            # Handle device mismatch for shuffling
            shuffled_samples = {}
            for k, v in samples.items():
                if k in ["hidden_states_combineds", "tembs"]:
                    # These tensors are on CPU, so use CPU permutation
                    shuffled_samples[k] = v[cpu_perm]
                else:
                    # Other tensors are on GPU, use GPU permutation
                    shuffled_samples[k] = v[perm]
            samples = shuffled_samples

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            pipeline.time_predictor.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                # Get actual number of timesteps from the sample data
                actual_num_timesteps = sample["latents"].shape[1]  # Maximum timesteps across batch
                timesteps_per_sample = sample["timesteps_per_sample"]  # Actual timesteps per sample
                train_timesteps = [step_index for step_index in range(actual_num_timesteps)]
                
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            # Compute diffusion model logprobs (only if not in time_predictor-only phase or if using joint training)
                            is_time_predictor_only_phase = epoch < config.train.get('time_predictor_only_epochs', 0)
                            
                            if not is_time_predictor_only_phase:
                                # Full joint training: compute both diffusion and time predictor logprobs
                                prev_sample, diffusion_log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            else:
                                # Time predictor only: skip diffusion logprob computation for efficiency
                                diffusion_log_prob = torch.zeros_like(sample["log_probs"][:, j])
                                prev_sample_mean = None
                                std_dev_t = None
                                prev_sample_mean_ref = None
                            
                            # Always compute time predictor logprobs (this is what we're training)
                            time_predictor_log_prob = compute_time_predictor_log_prob(pipeline, sample, j, embeds, pooled_embeds, config)
                            
                            # Compute time predictor KL divergence for regularization
                            time_predictor_kl_div = compute_time_predictor_kl_divergence(pipeline, sample, j, embeds, pooled_embeds, config)

                        # Create mask for active samples at this timestep
                        active_mask = (j < timesteps_per_sample).float()  # Shape: (batch_size,)
                        
                        # Combine logprobs: diffusion logprobs + time predictor logprobs
                        if is_time_predictor_only_phase:
                            # In time_predictor-only phase: only use time predictor logprobs
                            # Since diffusion model is frozen, diffusion logprobs cancel out in the ratio
                            current_log_prob = time_predictor_log_prob
                            reference_log_prob = sample["time_predictor_log_probs"][:, j] 
                        else:
                            # In joint training: combine both logprobs
                            current_log_prob = diffusion_log_prob + time_predictor_log_prob
                            reference_log_prob = sample["log_probs"][:, j] + sample["time_predictor_log_probs"][:, j]
                        
                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(current_log_prob - reference_log_prob)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        
                        # Apply mask to set inactive entries to 0, then divide by timesteps per sample and take mean
                        sample_losses = torch.maximum(unclipped_loss, clipped_loss) * active_mask
                        # Element-wise division by timesteps per sample, then batch mean
                        policy_loss = torch.mean(sample_losses / timesteps_per_sample.float())
                        
                        # Compute KL losses
                        loss = policy_loss
                        
                        # Add diffusion model KL regularization (only in joint training phase)
                        if config.train.beta > 0 and not is_time_predictor_only_phase:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_sample_losses = kl_loss.squeeze() * active_mask
                            # Element-wise division by timesteps per sample, then batch mean
                            diffusion_kl_loss = torch.mean(kl_sample_losses / timesteps_per_sample.float())
                            loss = loss + config.train.beta * diffusion_kl_loss
                        
                        # Add time predictor KL regularization (always active when training time predictor)
                        # Configuration: Set config.train.time_predictor_kl_weight = 0.01 (or desired value) 
                        # to control the strength of time predictor KL regularization
                        time_predictor_kl_weight = getattr(config.train, 'time_predictor_kl_weight', 0.01)  # Default weight
                        if time_predictor_kl_weight > 0:
                            time_predictor_kl_sample_losses = time_predictor_kl_div * active_mask
                            time_predictor_kl_loss = torch.mean(time_predictor_kl_sample_losses / timesteps_per_sample.float())
                            loss = loss + time_predictor_kl_weight * time_predictor_kl_loss

                        # Apply mask first, then compute mean over active entries only
                        masked_log_prob_diff = (current_log_prob - reference_log_prob) ** 2 * active_mask
                        info["approx_kl"].append(
                            0.5 * torch.sum(masked_log_prob_diff) / torch.sum(active_mask) if torch.sum(active_mask) > 0 else torch.tensor(0.0, device=active_mask.device)
                        )
                        
                        masked_clipfrac = ((torch.abs(ratio - 1.0) > config.train.clip_range).float() * active_mask)
                        info["clipfrac"].append(
                            torch.sum(masked_clipfrac) / torch.sum(active_mask) if torch.sum(active_mask) > 0 else torch.tensor(0.0, device=active_mask.device)
                        )
                        
                        masked_clipfrac_gt_one = ((ratio - 1.0 > config.train.clip_range).float() * active_mask)
                        info["clipfrac_gt_one"].append(
                            torch.sum(masked_clipfrac_gt_one) / torch.sum(active_mask) if torch.sum(active_mask) > 0 else torch.tensor(0.0, device=active_mask.device)
                        )
                        
                        masked_clipfrac_lt_one = ((1.0 - ratio > config.train.clip_range).float() * active_mask)
                        info["clipfrac_lt_one"].append(
                            torch.sum(masked_clipfrac_lt_one) / torch.sum(active_mask) if torch.sum(active_mask) > 0 else torch.tensor(0.0, device=active_mask.device)
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0 and not is_time_predictor_only_phase:
                            info["diffusion_kl_loss"].append(diffusion_kl_loss)
                        
                        # Always log time predictor KL loss when weight > 0
                        if time_predictor_kl_weight > 0:
                            info["time_predictor_kl_loss"].append(time_predictor_kl_loss)
                            info["time_predictor_kl_div_mean"].append(time_predictor_kl_div.mean())
                            info["time_predictor_kl_div_max"].append(time_predictor_kl_div.max())
                        
                        # Track separate logprob components for debugging
                        if not is_time_predictor_only_phase:
                            info["diffusion_log_prob_mean"].append(diffusion_log_prob.mean())
                        info["time_predictor_log_prob_mean"].append(time_predictor_log_prob.mean())
                        info["combined_log_prob_mean"].append(current_log_prob.mean())

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            current_trainable = get_current_trainable_parameters()
                            accelerator.clip_grad_norm_(
                                current_trainable, config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        # Handle different types of values in info dict
                        processed_info = {}
                        for k, v in info.items():
                            if k == "time_predictor_only_phase":
                                # Boolean values - just take the first (they should all be the same)
                                processed_info[k] = v[0] if v else False
                            else:
                                # Tensor values - compute mean
                                processed_info[k] = torch.mean(torch.stack(v))
                        
                        processed_info = accelerator.reduce(processed_info, reduction="mean")
                        processed_info.update({
                            "epoch": epoch, 
                            "inner_epoch": inner_epoch,
                            "time_predictor_only_phase": is_time_predictor_only_phase
                        })
                        if accelerator.is_main_process:
                            wandb.log(processed_info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    # Update EMA with current trainable parameters
                    current_trainable = get_current_trainable_parameters()
                    ema.step(current_trainable, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
        # Memory cleanup at end of epoch
        cleanup_memory()
        
if __name__ == "__main__":
    app.run(main)

