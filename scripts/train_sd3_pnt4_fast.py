from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import gc
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from omegaconf import OmegaConf, DictConfig
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../TPDM/src")))
from models.stable_diffusion_3.modeling_sd3_pnt import init_time_predictor, reshape_hidden_states_to_2d
from models.reference_distributions import get_ref_beta
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pnt_pipeline_with_logprob_fast import (
    pipeline_with_logprob,
    _decode_latents_for_time_predictor,
)
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
from typing import List

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

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config, per_step_active_mask=None):
    current_batch_size = sample["latents"].shape[0]
    device = sample["latents"].device

    # Gather tensors for the current and next timesteps
    current_latents = sample["latents"][:, j].to(device)
    next_latents = sample["latents"][:, j + 1].to(device)
    current_timesteps = sample["timesteps"][:, j].to(device)
    current_sigmas = sample["sigmas"][:, j].to(device)
    next_sigmas_target = sample["sigmas"][:, j + 1].to(device)

    # Prepare scheduler state expected by sde_step_with_logprob
    pipeline.scheduler.index_for_timestep = [{} for _ in range(current_batch_size)]
    n = 2
    for batch_idx in range(current_batch_size):
        pipeline.scheduler.index_for_timestep[batch_idx][current_timesteps[batch_idx].item()] = n

    max_step = max(n + 1, 1)
    pipeline.scheduler.sigmas = [torch.zeros(current_batch_size, device=device) for _ in range(max_step + 1)]
    pipeline.scheduler.sigmas[n] = current_sigmas
    pipeline.scheduler.sigmas[1] = sample["sigma_max"].to(device)

    # Handle per-step active mask
    if per_step_active_mask is None:
        per_step_active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=device)
    else:
        per_step_active_mask = per_step_active_mask.to(device)

    # Safety checks for transformer inputs

    # Forward pass through the diffusion transformer, retaining auxiliary features for the time predictor
    if config.train.cfg:
        transformer_outputs = transformer(
            hidden_states=torch.cat([current_latents] * 2),
            timestep=torch.cat([current_timesteps] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )
    else:
        transformer_outputs = transformer(
            hidden_states=current_latents,
            timestep=current_timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )

    noise_pred, temb, hidden_states_1, hidden_states_2 = transformer_outputs

    if config.train.cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

        temb_uncond, temb_text = temb.chunk(2)
        temb = temb_uncond + config.sample.guidance_scale * (temb_text - temb_uncond)

        hidden_states_1_uncond, hidden_states_1_text = hidden_states_1.chunk(2)
        hidden_states_1 = hidden_states_1_uncond + config.sample.guidance_scale * (
            hidden_states_1_text - hidden_states_1_uncond
        )
        hidden_states_2_uncond, hidden_states_2_text = hidden_states_2.chunk(2)
        hidden_states_2 = hidden_states_2_uncond + config.sample.guidance_scale * (
            hidden_states_2_text - hidden_states_2_uncond
        )

    # Prepare inputs for the time predictor
    if getattr(pipeline, "uses_image_time_predictor", False):
        time_predictor_inputs = _decode_latents_for_time_predictor(pipeline, current_latents)
    else:
        hidden_states_1 = reshape_hidden_states_to_2d(hidden_states_1)
        hidden_states_2 = reshape_hidden_states_to_2d(hidden_states_2)
        time_predictor_inputs = torch.cat([hidden_states_1, hidden_states_2], dim=1)

    if pipeline.use_vit_predictor:
        prompt_embeds_for_time_predictor = sample["prompt_embeds"]
        if prompt_embeds_for_time_predictor.device != device:
            prompt_embeds_for_time_predictor = prompt_embeds_for_time_predictor.to(
                device, dtype=time_predictor_inputs.dtype, non_blocking=True
            )
        else:
            prompt_embeds_for_time_predictor = prompt_embeds_for_time_predictor.to(
                dtype=time_predictor_inputs.dtype
            )
        time_preds = pipeline.time_predictor(time_predictor_inputs, temb, prompt_embeds_for_time_predictor)
    else:
        time_preds = pipeline.time_predictor(time_predictor_inputs, temb)

    sigma_next_pred = current_sigmas.clone()
    time_predictor_log_probs = torch.zeros_like(current_sigmas)

    for idx, (param1, param2) in enumerate(time_preds):
        if (not per_step_active_mask[idx]) or (current_sigmas[idx] < pipeline.min_sigma):
            zero_logprob = param1 * 0.0
            time_predictor_log_probs[idx] = zero_logprob
            sigma_next_pred[idx] = current_sigmas[idx]
            continue

        if pipeline.prediction_type == "alpha_beta":
            alpha, beta = param1, param2
        elif pipeline.prediction_type == "mode_concentration":
            alpha = param1 * (param2 - 2) + 1
            beta = (1 - param1) * (param2 - 2) + 1
        else:
            raise ValueError(f"Unsupported prediction type: {pipeline.prediction_type}")

        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)

        beta_dist = torch.distributions.Beta(alpha, beta)

        if pipeline.relative:
            valid_sigma = current_sigmas[idx].abs().clamp_min(pipeline.epsilon)
            ratio_target = (next_sigmas_target[idx] / valid_sigma).clamp(
                pipeline.epsilon, 1 - pipeline.epsilon
            )
        else:
            ratio_target = (current_sigmas[idx] - next_sigmas_target[idx]).clamp(
                pipeline.epsilon, 1 - pipeline.epsilon
            )

        ratio_pred = torch.clamp(beta_dist.mean, min=pipeline.epsilon, max=1 - pipeline.epsilon)
        ratio = ratio_pred + (ratio_target - ratio_pred).detach()

        if pipeline.relative:
            sigma_next_pred[idx] = current_sigmas[idx] * ratio
        else:
            sigma_next_pred[idx] = current_sigmas[idx] - ratio

        time_predictor_log_probs[idx] = beta_dist.log_prob(ratio_target)

    pipeline.scheduler.sigmas[n + 1] = sigma_next_pred

    # Debug: Check noise prediction for NaN/Inf

    # Prepare inputs for diffusion logprob computation
    current_latents_float = current_latents.float()
    next_latents_float = next_latents.float()

    prev_sample, diffusion_log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        current_timesteps,
        current_latents_float,
        prev_sample=next_latents_float,
        noise_level=config.sample.noise_level,
        active_mask=per_step_active_mask,
        sde_type=getattr(config.sample, "sde_type", "sde"),
    )

    return prev_sample, diffusion_log_prob, time_predictor_log_probs, prev_sample_mean, std_dev_t


def compute_time_predictor_log_prob_from_cache(
    pipeline,
    sample,
    j,
    config,
    per_step_active_mask=None,
):
    device = sample["latents"].device
    dtype = next(pipeline.time_predictor.parameters()).dtype

    current_sigmas = sample["sigmas"][:, j].to(device)
    next_sigmas_target = sample["sigmas"][:, j + 1].to(device)

    hidden_states_combined = sample["hidden_states_combineds"][:, j]
    if hidden_states_combined.device != device:
        hidden_states_combined = hidden_states_combined.to(device, dtype=dtype, non_blocking=True)
    else:
        hidden_states_combined = hidden_states_combined.to(dtype=dtype)

    temb = sample["tembs"][:, j]
    if temb.device != device:
        temb = temb.to(device, dtype=dtype, non_blocking=True)
    else:
        temb = temb.to(dtype=dtype)

    if pipeline.use_vit_predictor:
        prompt_embeds_for_time_predictor = sample["prompt_embeds"]
        if prompt_embeds_for_time_predictor.device != device:
            prompt_embeds_for_time_predictor = prompt_embeds_for_time_predictor.to(
                device, dtype=dtype, non_blocking=True
            )
        else:
            prompt_embeds_for_time_predictor = prompt_embeds_for_time_predictor.to(dtype=dtype)
        time_preds = pipeline.time_predictor(hidden_states_combined, temb, prompt_embeds_for_time_predictor)
    else:
        time_preds = pipeline.time_predictor(hidden_states_combined, temb)

    if per_step_active_mask is None:
        per_step_active_mask = torch.ones(current_sigmas.shape[0], dtype=torch.bool, device=device)
    else:
        per_step_active_mask = per_step_active_mask.to(device)

    time_predictor_log_probs = torch.zeros_like(current_sigmas)

    for idx, (param1, param2) in enumerate(time_preds):
        if (not per_step_active_mask[idx]) or (current_sigmas[idx] < pipeline.min_sigma):
            zero_logprob = param1 * 0.0
            time_predictor_log_probs[idx] = zero_logprob
            continue

        if pipeline.prediction_type == "alpha_beta":
            alpha, beta = param1, param2
        elif pipeline.prediction_type == "mode_concentration":
            alpha = param1 * (param2 - 2) + 1
            beta = (1 - param1) * (param2 - 2) + 1
        else:
            raise ValueError(f"Unsupported prediction type: {pipeline.prediction_type}")

        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)

        beta_dist = torch.distributions.Beta(alpha, beta)

        if pipeline.relative:
            valid_sigma = current_sigmas[idx].abs().clamp_min(pipeline.epsilon)
            ratio_target = (next_sigmas_target[idx] / valid_sigma).clamp(
                pipeline.epsilon, 1 - pipeline.epsilon
            )
        else:
            ratio_target = (current_sigmas[idx] - next_sigmas_target[idx]).clamp(
                pipeline.epsilon, 1 - pipeline.epsilon
            )

        ratio_pred = torch.clamp(beta_dist.mean, min=pipeline.epsilon, max=1 - pipeline.epsilon)
        ratio = ratio_pred + (ratio_target - ratio_pred).detach()

        time_predictor_log_probs[idx] = beta_dist.log_prob(ratio_target)

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
    # Optimize tensor transfers similar to compute_log_prob
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
    if pipeline.use_vit_predictor:
        time_preds = pipeline.time_predictor(hidden_states_combined, temb, embeds)
    else:
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
                images, _, _, _, _, _, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    mini_num_image_per_prompt=1,
                    guidance_scale=config.sample.eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                    process_index=accelerator.process_index,
                    sample_num_steps=config.sample.eval_num_steps,
                    sde_window_size=0,
                    sde_type=getattr(config.sample, "sde_type", "sde"),
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
    config.sample.sde_type = getattr(config.sample, "sde_type", "sde")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    if getattr(config.sample, "sde_window_size", 0) > 0:
        num_train_timesteps = config.sample.sde_window_size
    elif hasattr(config.sample, "train_num_steps"):
        num_train_timesteps = config.sample.train_num_steps
    else:
        num_train_timesteps = config.sample.num_steps - 1
        config.sample.train_num_steps = num_train_timesteps

    num_train_timesteps = max(int(num_train_timesteps), 1)
    config.sample.train_num_steps = num_train_timesteps

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # Check if we need special DDP handling for time_predictor-only training
    time_predictor_only_epochs = getattr(config.train, 'time_predictor_only_epochs', 0)
    freeze_time_predictor = getattr(config.train, "freeze_time_predictor", False)
    if freeze_time_predictor and time_predictor_only_epochs > 0:
        logger.warning(
            "freeze_time_predictor=True detected; overriding time_predictor_only_epochs to 0"
        )
        config.train.time_predictor_only_epochs = 0
        time_predictor_only_epochs = 0
    
    # Prepare kwargs for Accelerator initialization
    accelerator_kwargs = {
        "mixed_precision": config.mixed_precision,
        "project_config": accelerator_config,
        # IMPORTANT:
        # Gradient accumulation is counted per *outer* training iteration (one `accelerator.accumulate(...)` block).
        # In this script's training loop we accumulate once per trajectory/batch, not once per timestep.
        # Therefore we should NOT multiply by `num_train_timesteps` here.
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
    }
    
    # Determine if we need to enable find_unused_parameters for DDP
    enable_find_unused = (
        time_predictor_only_epochs > 0
        or getattr(config.train, "find_unused_parameters", False)
        or getattr(config, "require_find_unused_parameters", False)
        or getattr(config, "use_vit_predictor", False)
    )

    if enable_find_unused:
        from accelerate.utils import DistributedDataParallelKwargs

        existing_handlers = list(accelerator_kwargs.get("kwargs_handlers", []))
        existing_handlers.append(
            DistributedDataParallelKwargs(find_unused_parameters=True)
        )
        accelerator_kwargs["kwargs_handlers"] = existing_handlers

    # Ensure any prepared kwargs (e.g. DDP find_unused_parameters handler) are passed
    # into Accelerator. We also make certain the gradient_accumulation_steps matches
    # the intended inner training steps (uses config.sample.train_num_steps here).
    accelerator_kwargs["mixed_precision"] = config.mixed_precision
    accelerator_kwargs["project_config"] = accelerator_config
    accelerator_kwargs["gradient_accumulation_steps"] = (
        config.train.gradient_accumulation_steps
    )

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

    time_predictor_cfg = None
    config_path = getattr(config, "time_predictor_config_path", None)
    if config_path:
        resolved_path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Time predictor config not found: {resolved_path}")
        yaml_cfg = OmegaConf.load(resolved_path)
        if "time_predictor_config" in yaml_cfg:
            time_predictor_cfg = yaml_cfg["time_predictor_config"]
        else:
            time_predictor_cfg = yaml_cfg

    use_image_time_predictor = getattr(config, "use_image_time_predictor", False)
    if use_image_time_predictor and not getattr(config, "use_vit_predictor", False):
        raise ValueError("Image-based time predictor requires use_vit_predictor=True")

    if use_image_time_predictor:
        if time_predictor_cfg is None:
            raise ValueError(
                "use_image_time_predictor=True requires a time_predictor_config_path with decoded_image settings"
            )
        image_size_override = getattr(config, "time_predictor_image_size", None)
        if isinstance(time_predictor_cfg, DictConfig):
            time_predictor_cfg.input_type = "decoded_image"
            if not time_predictor_cfg.get("in_channels", None):
                time_predictor_cfg.in_channels = 3
            if image_size_override is not None and not time_predictor_cfg.get("image_size", None):
                time_predictor_cfg.image_size = image_size_override
        else:
            setattr(time_predictor_cfg, "input_type", "decoded_image")
            if getattr(time_predictor_cfg, "in_channels", None) is None:
                setattr(time_predictor_cfg, "in_channels", 3)
            if image_size_override is not None and getattr(time_predictor_cfg, "image_size", None) is None:
                setattr(time_predictor_cfg, "image_size", image_size_override)

    init_time_predictor(
        pipeline,
        config.sd3_checkpoint_path,
        use_vit_predictor=config.use_vit_predictor,
        time_predictor_config=time_predictor_cfg,
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.time_predictor.requires_grad_(not freeze_time_predictor)
    if freeze_time_predictor:
        logger.info("Time predictor freezing enabled; gradients are disabled.")

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
    time_predictor_parameters = [p for p in pipeline.time_predictor.parameters() if p.requires_grad]
    all_trainable_parameters = transformer_trainable_parameters + time_predictor_parameters
    # This ema setting affects the previous 2 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(all_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc
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
        # For fast mode we repeat mini images inside the pipeline; sampler should operate on
        # the effective train batch size (config.train.batch_size) and pick k as the
        # number of unique samples per batch: num_image_per_prompt // mini_num_image_per_prompt
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt // config.sample.mini_num_image_per_prompt,
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
            k=config.sample.num_image_per_prompt // config.sample.mini_num_image_per_prompt,
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
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size*config.sample.mini_num_image_per_prompt, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size*config.sample.mini_num_image_per_prompt, 1)

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
        time_predictor_only_epochs = 0 if freeze_time_predictor else config.train.time_predictor_only_epochs
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
                    images, latents, log_probs, time_predictor_log_probs, timesteps, sigma_max, all_sigmas_per_step, hidden_states_combineds, tembs, step_counts, all_active_masks = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        mini_num_image_per_prompt=config.sample.mini_num_image_per_prompt,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        train_num_steps=config.sample.train_num_steps,
                        process_index=accelerator.process_index,
                        sample_num_steps=config.sample.num_steps,
                        generator=generator,
                        sde_window_size=config.sample.sde_window_size,
                        sde_window_range=config.sample.sde_window_range,
                        sde_type=config.sample.sde_type,
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            time_predictor_log_probs = torch.stack(time_predictor_log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            # Stack sigmas to match timesteps and latents structure
            # all_sigmas_per_step contains sigma values for each step, we need to stack them
            sigmas = torch.stack(all_sigmas_per_step, dim=1)  # (batch_size, num_steps + 1)

            timesteps = torch.stack(timesteps, dim=1)
            # compute rewards asynchronously
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids.repeat(config.sample.mini_num_image_per_prompt,1), skip_special_tokens=True
            )

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids.repeat(config.sample.mini_num_image_per_prompt,1),
                    "prompt_embeds": prompt_embeds.repeat(config.sample.mini_num_image_per_prompt,1,1),
                    "pooled_prompt_embeds": pooled_prompt_embeds.repeat(config.sample.mini_num_image_per_prompt,1),
                    "timesteps": timesteps,
                    "latents": latents,  # Store full latents tensor (no slicing to save memory)
                    # Note: next_latents removed - will use latents[:, j+1] when needed
                    "log_probs": log_probs,
                    "time_predictor_log_probs": time_predictor_log_probs,
                    "hidden_states_combineds": hidden_states_combineds,
                    "tembs": tembs,
                    "rewards": rewards,
                    "sigmas": sigmas,  # sigma values for each timestep (needs num_steps + 1 for next_sigma access)
                    "sigma_max": sigma_max,
                    "step_counts": step_counts,
                    "active_masks": all_active_masks,
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
        
        # Clean up large tensors immediately
        del latents, log_probs, time_predictor_log_probs, sigmas, timesteps
        gc.collect()

        # Pad tensors to the same length before collation to handle variable timesteps
        # Note: latents now has shape (batch_size, num_steps + 1) 
        # while log_probs have (batch_size, num_steps)
        # sigmas has shape (batch_size, num_steps + 1) like latents
        max_timesteps_latents = max(s["latents"].shape[1] for s in samples)  # This is num_steps + 1
        max_timesteps_logprobs = max_timesteps_latents - 1  # This is num_steps for log_probs
        
        for sample in samples:
            current_timesteps_latents = sample["latents"].shape[1]
            if current_timesteps_latents < max_timesteps_latents:
                
                # Use more memory-efficient padding by pre-allocating full-size tensors
                # and copying data instead of concatenating
                # Handle latents - it already has num_steps + 1 timesteps
                if "latents" in sample:
                    original_tensor = sample["latents"]
                    actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                    # For latents, pad to max_timesteps_latents
                    full_shape = [original_tensor.shape[0], max_timesteps_latents] + list(original_tensor.shape[2:])
                    new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                    # Copy original data using actual tensor dimensions
                    new_tensor[:, :actual_timesteps] = original_tensor
                    # Fill padding with last latent value
                    if actual_timesteps > 0:
                        last_latent = original_tensor[:, -1:]
                        pad_size_actual = max_timesteps_latents - actual_timesteps
                        new_tensor[:, actual_timesteps:] = last_latent.repeat(1, pad_size_actual, 1, 1, 1)
                    sample["latents"] = new_tensor
                    del original_tensor  # Explicit cleanup
                
                # Same for log_probs and time_predictor_log_probs (they have num_steps)
                for logprob_key in ["log_probs", "time_predictor_log_probs"]:
                    if logprob_key in sample:
                        original_tensor = sample[logprob_key]
                        actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                        full_shape = [original_tensor.shape[0], max_timesteps_logprobs]
                        new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                        new_tensor[:, :actual_timesteps] = original_tensor
                        sample[logprob_key] = new_tensor
                        del original_tensor  # Explicit cleanup
                
                # More memory-efficient padding for hidden_states_combineds and tembs (they have num_steps)
                for tensor_key in ["hidden_states_combineds", "tembs"]:
                    if tensor_key in sample:
                        original_tensor = sample[tensor_key]
                        actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                        full_shape = [original_tensor.shape[0], max_timesteps_logprobs] + list(original_tensor.shape[2:])
                        new_tensor = torch.zeros(full_shape, device=original_tensor.device, dtype=original_tensor.dtype)
                        new_tensor[:, :actual_timesteps] = original_tensor
                        sample[tensor_key] = new_tensor
                        del original_tensor  # Explicit cleanup
                
                # Pad timesteps - use the last timestep value for padding (they have num_steps)
                if "timesteps" in sample:
                    original_tensor = sample["timesteps"]
                    actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                    new_tensor = torch.zeros([original_tensor.shape[0], max_timesteps_logprobs], 
                                           device=original_tensor.device, dtype=original_tensor.dtype)
                    new_tensor[:, :actual_timesteps] = original_tensor
                    # Fill padding with last timestep value
                    if actual_timesteps > 0:
                        last_timestep = original_tensor[:, -1:]
                        pad_size_actual = max_timesteps_logprobs - actual_timesteps
                        new_tensor[:, actual_timesteps:] = last_timestep.repeat(1, pad_size_actual)
                    sample["timesteps"] = new_tensor
                    del original_tensor  # Explicit cleanup
                
                # Pad sigmas - they have num_steps + 1 elements like latents
                if "sigmas" in sample:
                    original_tensor = sample["sigmas"]
                    actual_timesteps = original_tensor.shape[1]  # Use actual tensor dimension
                    new_tensor = torch.zeros([original_tensor.shape[0], max_timesteps_latents], 
                                           device=original_tensor.device, dtype=original_tensor.dtype)
                    new_tensor[:, :actual_timesteps] = original_tensor
                    # Fill padding with last sigma value
                    if actual_timesteps > 0:
                        last_sigma = original_tensor[:, -1:]
                        pad_size_actual = max_timesteps_latents - actual_timesteps
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

        # Log average active denoising steps per sample
        step_counts_tensor = samples["step_counts"].to(accelerator.device, dtype=torch.float32)
        gathered_step_counts = accelerator.gather(step_counts_tensor)
        if accelerator.is_main_process:
            wandb.log(
                {"avg_active_steps": gathered_step_counts.mean().item()},
                step=global_step,
            )
        del samples["step_counts"]

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
        # Get maximum padded timesteps and actual timesteps per sample
        # latents has shape (batch_size, num_steps + 1), but rewards need to match timesteps (num_steps)
        max_padded_timesteps_rewards = samples["latents"].shape[1] - 1  # Subtract 1 for rewards
        num_train_steps = num_train_timesteps  # Fixed number of training steps in fast version
        
        # Apply gamma discounting like in modeling_sd3_pnt.py reward function
        # Only apply during time predictor only training phase
        gamma = config.reward_gamma
        if is_time_predictor_only_phase and gamma < 1.0:  # Only apply gamma discounting during time predictor only training
            # Create gamma-discounted rewards for each timestep using fixed train steps
            batch_size = samples["rewards"]["avg"].shape[0]
            discounted_rewards = torch.zeros(batch_size, max_padded_timesteps_rewards, device=samples["rewards"]["avg"].device)
            
            for i in range(batch_size):
                final_reward = samples["rewards"]["avg"][i].item()
                
                # Apply gamma discounting: reward_t = final_reward * gamma^(last_timestep - t)
                for t in range(num_train_steps):
                    discounted_rewards[i, t] = final_reward * (gamma ** (num_train_steps - 1 - t))
                # Normalize by the actual number of timesteps (like in modeling_sd3_pnt.py)
                discounted_rewards[i, :num_train_steps] = discounted_rewards[i, :num_train_steps] / num_train_steps
                # Padded timesteps remain 0
            
            samples["rewards"]["avg"] = discounted_rewards
        else:
            # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
            samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, max_padded_timesteps_rewards)
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
        if true_count % num_batches != 0 or true_count == 0:
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
                
                batch_size = sample["latents"].shape[0]
                log_dtype = sample["log_probs"].dtype
                device = accelerator.device

                def _get_per_step_active_mask(sample_dict, step_idx: int) -> torch.Tensor:
                    if "active_masks" in sample_dict and sample_dict["active_masks"].numel() != 0:
                        return sample_dict["active_masks"][:, step_idx].to(accelerator.device)
                    return torch.ones(sample_dict["latents"].shape[0], dtype=torch.bool, device=accelerator.device)

                # ------------------------------
                # Two-pass objective to avoid retaining activations over all timesteps.
                # Pass 1 (no_grad): compute trajectory-level ratio/clipping decision + logging stats.
                # Pass 2 (grad): per-timestep backward of a surrogate whose gradient matches PPO w/ trajectory ratio.
                # ------------------------------

                with accelerator.accumulate(transformer):
                    # Save RNG state so the two-pass recomputation uses identical stochastic masks
                    # (e.g., dropout) in pass-1 and pass-2. Without this, ratio/clipping decisions can be
                    # computed from different stochastic forward outputs than the ones used for gradients,
                    # which increases variance and can look like numerical instability/collapse.
                    rng_state_cpu = torch.random.get_rng_state()
                    rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                    # ---- Pass 1: compute ratio and masks without gradient ----
                    # Use the same autocast policy as the gradient pass to keep numerics consistent.
                    with torch.no_grad(), autocast():
                        current_log_prob_sum_ng = torch.zeros(batch_size, device=device, dtype=log_dtype)
                        reference_log_prob_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        advantages_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        active_mask_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        log_prob_diff_sq_sum_ng = torch.zeros_like(current_log_prob_sum_ng)

                        diffusion_log_prob_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        time_predictor_log_prob_sum_ng = torch.zeros_like(current_log_prob_sum_ng)

                        diffusion_kl_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        time_predictor_kl_sum_ng = torch.zeros_like(current_log_prob_sum_ng)
                        time_predictor_kl_max_ng = torch.full(
                            (batch_size,), float("-inf"), device=device, dtype=log_dtype
                        )

                        for j in range(num_train_timesteps):
                            per_step_active_mask = _get_per_step_active_mask(sample, j)
                            active_mask_float = per_step_active_mask.to(dtype=log_dtype)

                            if not is_time_predictor_only_phase:
                                (
                                    _,
                                    diffusion_log_prob_ng,
                                    time_predictor_log_prob_ng,
                                    prev_sample_mean_ng,
                                    std_dev_t_ng,
                                ) = compute_log_prob(
                                    transformer,
                                    pipeline,
                                    sample,
                                    j,
                                    embeds,
                                    pooled_embeds,
                                    config,
                                    per_step_active_mask,
                                )

                                prev_sample_mean_ref_ng = None
                                if config.train.beta > 0:
                                    with transformer.module.disable_adapter():
                                        (
                                            _,
                                            _,
                                            _,
                                            prev_sample_mean_ref_ng,
                                            _,
                                        ) = compute_log_prob(
                                            transformer,
                                            pipeline,
                                            sample,
                                            j,
                                            embeds,
                                            pooled_embeds,
                                            config,
                                            per_step_active_mask,
                                        )
                            else:
                                time_predictor_log_prob_ng = compute_time_predictor_log_prob_from_cache(
                                    pipeline,
                                    sample,
                                    j,
                                    config,
                                    per_step_active_mask,
                                )
                                diffusion_log_prob_ng = torch.zeros_like(sample["log_probs"][:, j])
                                prev_sample_mean_ng = None
                                std_dev_t_ng = None
                                prev_sample_mean_ref_ng = None

                            if freeze_time_predictor:
                                time_predictor_log_prob_ng = torch.zeros_like(sample["log_probs"][:, j])
                                time_predictor_kl_div_ng = None
                            else:
                                if config.train.time_predictor_kl_weight > 0:
                                    time_predictor_kl_div_ng = compute_time_predictor_kl_divergence(
                                        pipeline, sample, j, embeds, pooled_embeds, config
                                    )
                                else:
                                    time_predictor_kl_div_ng = None

                            # Combine logprobs as in the original objective.
                            if is_time_predictor_only_phase:
                                current_log_prob_ng = time_predictor_log_prob_ng
                                reference_log_prob_ng = sample["time_predictor_log_probs"][:, j]
                            elif freeze_time_predictor:
                                current_log_prob_ng = diffusion_log_prob_ng
                                reference_log_prob_ng = sample["log_probs"][:, j]
                            else:
                                current_log_prob_ng = diffusion_log_prob_ng + time_predictor_log_prob_ng
                                reference_log_prob_ng = sample["log_probs"][:, j] + sample["time_predictor_log_probs"][:, j]

                            advantages_ng = torch.clamp(
                                sample["advantages"][:, j].to(log_dtype),
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )

                            current_log_prob_sum_ng = current_log_prob_sum_ng + current_log_prob_ng * active_mask_float
                            reference_log_prob_sum_ng = reference_log_prob_sum_ng + reference_log_prob_ng * active_mask_float
                            advantages_sum_ng = advantages_sum_ng + advantages_ng * active_mask_float
                            active_mask_sum_ng = active_mask_sum_ng + active_mask_float
                            log_prob_diff_sq_sum_ng = log_prob_diff_sq_sum_ng + (
                                (current_log_prob_ng - reference_log_prob_ng) ** 2 * active_mask_float
                            )

                            if not is_time_predictor_only_phase:
                                diffusion_log_prob_sum_ng = diffusion_log_prob_sum_ng + diffusion_log_prob_ng * active_mask_float
                            if not freeze_time_predictor or is_time_predictor_only_phase:
                                time_predictor_log_prob_sum_ng = (
                                    time_predictor_log_prob_sum_ng + time_predictor_log_prob_ng * active_mask_float
                                )

                            if config.train.beta > 0 and not is_time_predictor_only_phase:
                                if prev_sample_mean_ref_ng is not None:
                                    kl_loss_ng = ((prev_sample_mean_ng - prev_sample_mean_ref_ng) ** 2).mean(
                                        dim=(1, 2, 3), keepdim=True
                                    ) / (2 * std_dev_t_ng ** 2)
                                    diffusion_kl_sum_ng = diffusion_kl_sum_ng + kl_loss_ng.squeeze() * active_mask_float

                            if (
                                config.train.time_predictor_kl_weight > 0
                                and not freeze_time_predictor
                                and time_predictor_kl_div_ng is not None
                            ):
                                time_predictor_kl_sum_ng = time_predictor_kl_sum_ng + time_predictor_kl_div_ng * active_mask_float
                                masked_tp_kl = torch.where(
                                    per_step_active_mask,
                                    time_predictor_kl_div_ng,
                                    torch.full_like(time_predictor_kl_div_ng, float("-inf")),
                                )
                                time_predictor_kl_max_ng = torch.maximum(time_predictor_kl_max_ng, masked_tp_kl)

                        log_prob_diff_sum = current_log_prob_sum_ng - reference_log_prob_sum_ng
                        ratio = torch.exp(log_prob_diff_sum)
                        ratio_clipped = torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )

                        advantages_scaled = advantages_sum_ng / float(num_train_steps)

                        unclipped_loss = -advantages_scaled * ratio
                        clipped_loss = -advantages_scaled * ratio_clipped
                        per_sample_policy_loss = torch.maximum(unclipped_loss, clipped_loss)
                        policy_loss_value = torch.mean(per_sample_policy_loss)

                        # For the clipped PPO objective, gradient is zero where the clipped branch is strictly selected.
                        grad_mask = (unclipped_loss >= clipped_loss).to(dtype=log_dtype)

                        # This coefficient multiplies sum_j ∇ log π_j.
                        # It is treated as a constant during backprop to avoid holding a full trajectory graph.
                        policy_grad_coef = (-advantages_scaled) * ratio * grad_mask

                        diffusion_kl_loss_value = None
                        if config.train.beta > 0 and not is_time_predictor_only_phase:
                            diffusion_kl_loss_value = torch.mean(diffusion_kl_sum_ng / float(num_train_steps))

                        time_predictor_kl_loss_value = None
                        if config.train.time_predictor_kl_weight > 0 and not freeze_time_predictor:
                            time_predictor_kl_loss_value = torch.mean(time_predictor_kl_sum_ng / float(num_train_steps))

                        total_active = torch.sum(active_mask_sum_ng)
                        if total_active <= 0:
                            total_active = torch.tensor(1.0, device=active_mask_sum_ng.device)

                        # Populate logging info from the no_grad pass.
                        info["approx_kl"].append(0.5 * torch.sum(log_prob_diff_sq_sum_ng) / total_active)
                        clip_mask = (torch.abs(ratio - 1.0) > config.train.clip_range).float() * active_mask_sum_ng
                        info["clipfrac"].append(torch.sum(clip_mask) / total_active)
                        clip_mask_gt = ((ratio - 1.0) > config.train.clip_range).float() * active_mask_sum_ng
                        info["clipfrac_gt_one"].append(torch.sum(clip_mask_gt) / total_active)
                        clip_mask_lt = ((1.0 - ratio) > config.train.clip_range).float() * active_mask_sum_ng
                        info["clipfrac_lt_one"].append(torch.sum(clip_mask_lt) / total_active)
                        info["policy_loss"].append(policy_loss_value)

                        if diffusion_kl_loss_value is not None:
                            info["diffusion_kl_loss"].append(diffusion_kl_loss_value)
                        if time_predictor_kl_loss_value is not None:
                            info["time_predictor_kl_loss"].append(time_predictor_kl_loss_value)
                            info["time_predictor_kl_div_mean"].append(torch.sum(time_predictor_kl_sum_ng) / total_active)
                            if (time_predictor_kl_max_ng > float("-inf")).any():
                                info["time_predictor_kl_div_max"].append(
                                    torch.where(
                                        time_predictor_kl_max_ng > float("-inf"),
                                        time_predictor_kl_max_ng,
                                        torch.zeros_like(time_predictor_kl_max_ng),
                                    ).max()
                                )
                            else:
                                info["time_predictor_kl_div_max"].append(
                                    torch.tensor(0.0, device=log_prob_diff_sq_sum_ng.device, dtype=log_dtype)
                                )
                        if not is_time_predictor_only_phase:
                            info["diffusion_log_prob_mean"].append(torch.sum(diffusion_log_prob_sum_ng) / total_active)
                        if not freeze_time_predictor or is_time_predictor_only_phase:
                            info["time_predictor_log_prob_mean"].append(torch.sum(time_predictor_log_prob_sum_ng) / total_active)
                        info["combined_log_prob_mean"].append(torch.sum(current_log_prob_sum_ng) / total_active)

                        loss_value = policy_loss_value
                        if diffusion_kl_loss_value is not None:
                            loss_value = loss_value + config.train.beta * diffusion_kl_loss_value
                        if time_predictor_kl_loss_value is not None:
                            loss_value = loss_value + config.train.time_predictor_kl_weight * time_predictor_kl_loss_value
                        info["loss"].append(loss_value)

                    # Restore RNG state so pass-2 sees the same stochastic forward masks as pass-1.
                    torch.random.set_rng_state(rng_state_cpu)
                    if rng_state_cuda is not None:
                        torch.cuda.set_rng_state_all(rng_state_cuda)

                    # ---- Pass 2: per-timestep backward without retaining full-trajectory activations ----
                    optimizer.zero_grad()
                    for j in tqdm(
                        range(num_train_timesteps),
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                    ):
                        per_step_active_mask = _get_per_step_active_mask(sample, j)
                        active_mask_float = per_step_active_mask.to(dtype=log_dtype)

                        with autocast():
                            if not is_time_predictor_only_phase:
                                (
                                    _,
                                    diffusion_log_prob,
                                    time_predictor_log_prob,
                                    prev_sample_mean,
                                    std_dev_t,
                                ) = compute_log_prob(
                                    transformer,
                                    pipeline,
                                    sample,
                                    j,
                                    embeds,
                                    pooled_embeds,
                                    config,
                                    per_step_active_mask,
                                )

                                prev_sample_mean_ref = None
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            (
                                                _,
                                                _,
                                                _,
                                                prev_sample_mean_ref,
                                                _,
                                            ) = compute_log_prob(
                                                transformer,
                                                pipeline,
                                                sample,
                                                j,
                                                embeds,
                                                pooled_embeds,
                                                config,
                                                per_step_active_mask,
                                            )
                            else:
                                time_predictor_log_prob = compute_time_predictor_log_prob_from_cache(
                                    pipeline,
                                    sample,
                                    j,
                                    config,
                                    per_step_active_mask,
                                )
                                diffusion_log_prob = torch.zeros_like(sample["log_probs"][:, j])
                                prev_sample_mean = None
                                std_dev_t = None
                                prev_sample_mean_ref = None

                            if freeze_time_predictor:
                                time_predictor_log_prob = torch.zeros_like(sample["log_probs"][:, j])
                                time_predictor_kl_div = None
                            else:
                                if config.train.time_predictor_kl_weight > 0:
                                    time_predictor_kl_div = compute_time_predictor_kl_divergence(
                                        pipeline, sample, j, embeds, pooled_embeds, config
                                    )
                                else:
                                    time_predictor_kl_div = None

                            # Combine logprobs as in the original objective.
                            if is_time_predictor_only_phase:
                                current_log_prob = time_predictor_log_prob
                            elif freeze_time_predictor:
                                current_log_prob = diffusion_log_prob
                            else:
                                current_log_prob = diffusion_log_prob + time_predictor_log_prob

                            # Surrogate whose gradient matches: ∇L = (-A_scaled) * mask * ∇r,
                            # with ∇r = r * Σ_j active_j ∇ log π_j.
                            policy_step_loss = torch.mean(
                                policy_grad_coef.detach() * (current_log_prob * active_mask_float)
                            )

                            step_loss = policy_step_loss

                            if config.train.beta > 0 and (not is_time_predictor_only_phase):
                                if prev_sample_mean_ref is not None:
                                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
                                        dim=(1, 2, 3), keepdim=True
                                    ) / (2 * std_dev_t ** 2)
                                    diffusion_kl_step = torch.mean((kl_loss.squeeze() * active_mask_float)) / float(
                                        num_train_steps
                                    )
                                    step_loss = step_loss + config.train.beta * diffusion_kl_step

                            if (
                                config.train.time_predictor_kl_weight > 0
                                and not freeze_time_predictor
                                and time_predictor_kl_div is not None
                            ):
                                tp_kl_step = torch.mean((time_predictor_kl_div * active_mask_float)) / float(
                                    num_train_steps
                                )
                                step_loss = step_loss + config.train.time_predictor_kl_weight * tp_kl_step

                        if torch.isnan(step_loss) or torch.isinf(step_loss):
                            logger.warning(
                                f"NaN/Inf detected in per-step loss at j={j}: {step_loss}. Skipping backward for this timestep."
                            )
                            continue

                        # IMPORTANT: When `accelerator.sync_gradients` is True, DDP would normally all-reduce
                        # gradients on *every* backward call. Since we do multiple backwards per batch (one per
                        # timestep), we explicitly suppress gradient synchronization for all but the last timestep.
                        should_suppress_sync = accelerator.sync_gradients and (j != num_train_timesteps - 1)
                        if should_suppress_sync:
                            with contextlib.ExitStack() as stack:
                                stack.enter_context(accelerator.no_sync(transformer))
                                if not freeze_time_predictor:
                                    stack.enter_context(accelerator.no_sync(pipeline.time_predictor))
                                accelerator.backward(step_loss)
                        else:
                            accelerator.backward(step_loss)

                    if accelerator.sync_gradients:
                        current_trainable = get_current_trainable_parameters()
                        accelerator.clip_grad_norm_(current_trainable, config.train.max_grad_norm)
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
                            "time_predictor_only_phase": is_time_predictor_only_phase,
                            "time_predictor_frozen": freeze_time_predictor,
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
