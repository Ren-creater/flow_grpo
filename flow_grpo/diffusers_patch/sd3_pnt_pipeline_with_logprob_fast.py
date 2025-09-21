# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_pnt_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
# - Adapted to work with SD3PredictNextTimeStepModel instead of StableDiffusion3Pipeline
from typing import Any, Dict, List, Optional, Union
import torch
import random
from .sd3_pnt_sde_with_logprob import sde_step_with_logprob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../TPDM/src")))
from models.stable_diffusion_3.modeling_sd3_pnt import reshape_hidden_states_to_2d

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    mini_num_image_per_prompt: int = 1,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    train_num_steps: int = 1,
    process_index: int = 0,
    sample_num_steps: int = 10,
    random_timestep: Optional[int] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    # 5. Initialize timestep prediction variables
    sigma = torch.ones(batch_size, dtype=latents.dtype, device=device)

    random.seed(process_index)
    if random_timestep is None:
        random_timestep = random.randint(0, sample_num_steps//2)

    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []
    all_time_predictor_log_probs = []
    all_timesteps = []
    all_hidden_states_combineds = []
    all_tembs = []
    all_sigmas_per_step = []
    all_active_masks = []
    
    # Clear and initialize scheduler for batched timesteps/sigmas
    self.scheduler.timesteps = []
    self.scheduler.sigmas = [sigma.clone()]
    self.scheduler.index_for_timestep = [{} for _ in range(batch_size)]

    if self.do_classifier_free_guidance:
        tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 7. Denoising loop with timestep prediction
    for step in range(num_inference_steps):
        if self.interrupt:
            continue

        # Determine noise level based on step position
        if step < random_timestep:
            cur_noise_level = 0
        elif step == random_timestep:
            cur_noise_level = noise_level
            # Repeat latents and embeddings for mini_num_image_per_prompt
            latents = latents.repeat(mini_num_image_per_prompt, 1, 1, 1)
            prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
            if self.do_classifier_free_guidance:
                tem_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                tem_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            # Update sigma to match new batch size
                sigma = sigma.repeat(mini_num_image_per_prompt)
                # Also repeat active_mask when repeating latents
                if 'active_mask' in locals():
                    active_mask = active_mask.repeat(mini_num_image_per_prompt)
                else:
                    active_mask = torch.ones_like(sigma, dtype=torch.bool)
            # Update scheduler sigmas to repeat previous sigmas
            self.scheduler.sigmas = [s.repeat(mini_num_image_per_prompt) for s in self.scheduler.sigmas]
            # Update scheduler for new batch size
            self.scheduler.index_for_timestep = [{} for _ in range(len(latents))]
            all_latents.append(latents)
        elif step > random_timestep and step < random_timestep + train_num_steps:
            cur_noise_level = noise_level
        else:
            cur_noise_level = 0

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = sigma * 1000  # Convert sigma to timestep-like values
        
        # Call transformer to get noise prediction and features for time predictor
        (noise_pred, temb, hidden_states_1, hidden_states_2) = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep.repeat(2) if self.do_classifier_free_guidance else timestep,
            encoder_hidden_states=tem_prompt_embeds,
            pooled_projections=tem_pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            temb_uncond, temb_text = temb.chunk(2)
            temb = temb_uncond + self.guidance_scale * (temb_text - temb_uncond)
            hidden_states_1_uncond, hidden_states_1_text = hidden_states_1.chunk(2)
            hidden_states_1 = hidden_states_1_uncond + self.guidance_scale * (
                hidden_states_1_text - hidden_states_1_uncond
            )
            hidden_states_2_uncond, hidden_states_2_text = hidden_states_2.chunk(2)
            hidden_states_2 = hidden_states_2_uncond + self.guidance_scale * (
                hidden_states_2_text - hidden_states_2_uncond
            )

        # Reshape and combine hidden states for time predictor
        hidden_states_1 = reshape_hidden_states_to_2d(hidden_states_1)
        hidden_states_2 = reshape_hidden_states_to_2d(hidden_states_2)
        hidden_states_combined = torch.cat([hidden_states_1, hidden_states_2], dim=1)
        
        # Store hidden states and temporal embeddings only for training window
        if step >= random_timestep and step < random_timestep + train_num_steps:
            all_hidden_states_combineds.append(hidden_states_combined.half().cpu())
            all_tembs.append(temb.half().cpu())

        # Predict next sigma using time predictor and collect logprobs
        if self.use_vit_predictor:
            time_preds = self.time_predictor(hidden_states_combined, temb, prompt_embeds)
        else:
            time_preds = self.time_predictor(hidden_states_combined, temb)
        current_batch_size = len(latents)
        sigma_next = torch.zeros_like(sigma)
        step_time_predictor_log_probs = torch.zeros_like(sigma)
        
        for i, (param1, param2) in enumerate(time_preds):
            if self.prediction_type == "alpha_beta":
                alpha, beta = param1, param2
            elif self.prediction_type == "mode_concentration":
                alpha = param1 * (param2 - 2) + 1
                beta = (1 - param1) * (param2 - 2) + 1
            beta_dist = torch.distributions.Beta(alpha, beta)

            ratio = beta_dist.sample()
            ratio = (
                ratio.clamp(self.epsilon, 1 - self.epsilon)
                if self.relative
                else ratio.clamp(self.epsilon, sigma[i]).clamp(0, 1 - self.epsilon)
            )
            sigma_next[i] = sigma[i] * ratio if self.relative else sigma[i] - ratio
            
            # Compute time predictor log probability for this ratio
            time_predictor_log_prob = beta_dist.log_prob(ratio)
            step_time_predictor_log_probs[i] = time_predictor_log_prob
            
            # Check if this sample should stop
            if sigma[i] < self.min_sigma or sigma_next[i] < self.min_sigma:
                sigma_next[i] = torch.tensor(0.0).to(sigma_next.device)
                # Mark as inactive for next steps
                if 'active_mask' in locals():
                    active_mask[i] = False
                else:
                    active_mask = torch.ones_like(sigma, dtype=torch.bool)
                    active_mask[i] = False

    # Update scheduler state for this timestep
        self.scheduler.timesteps.append(timestep.clone())
        self.scheduler.sigmas.append(sigma_next.clone())
        # Update each batch element's index_for_timestep dictionary
        for i, t in enumerate(timestep):
            self.scheduler.index_for_timestep[i][t.item()] = step

        latents_dtype = latents.dtype

        # Apply SDE step to all samples, passing active_mask for safety
        if 'active_mask' not in locals():
            active_mask = torch.ones(len(latents), dtype=torch.bool, device=latents.device)
        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            self.scheduler, 
            noise_pred.float(), 
            timestep,
            latents.float(),
            noise_level=cur_noise_level,
            active_mask=active_mask,
        )
        
        # Store results only for training window
        if step >= random_timestep and step < random_timestep + train_num_steps:
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_time_predictor_log_probs.append(step_time_predictor_log_probs)
            all_timesteps.append(timestep)
            all_sigmas_per_step.append(sigma.clone())
            # store active mask for this step so training can ignore inactive samples
            all_active_masks.append(active_mask.clone())
        elif step == random_timestep + train_num_steps:
            all_sigmas_per_step.append(sigma.clone())
            
        sigma = sigma_next

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    if hasattr(self, 'maybe_free_model_hooks'):
        self.maybe_free_model_hooks()

    # Stack hidden states and temporal embeddings to match timesteps structure
    if all_hidden_states_combineds:
        all_hidden_states_combineds = torch.stack(all_hidden_states_combineds, dim=1)
        all_tembs = torch.stack(all_tembs, dim=1)
    else:
        # If no training steps were collected, create empty tensors
        all_hidden_states_combineds = torch.empty(0)
        all_tembs = torch.empty(0)

    # Provide sigma_max repeated to mini_num_image_per_prompt
    base_sigma_max = self.scheduler.sigmas[1].clone() if len(self.scheduler.sigmas) > 1 else sigma.clone()
    sigma_max = base_sigma_max.repeat(mini_num_image_per_prompt) if base_sigma_max.dim() == 0 or len(base_sigma_max) == 1 else base_sigma_max

    # Stack/return active masks aligned with all_latents/all_sigmas_per_step
    if all_active_masks:
        all_active_masks = torch.stack(all_active_masks, dim=1)
    else:
        all_active_masks = torch.empty(0)

    return (
        image,
        all_latents,
        all_log_probs,
        all_time_predictor_log_probs,
        all_timesteps,
        sigma_max,
        all_sigmas_per_step,
        all_hidden_states_combineds,
        all_tembs,
        all_active_masks,
    )