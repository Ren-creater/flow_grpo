# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    active_mask: Optional[torch.BoolTensor] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    # Handle inactive samples - preserve original sample for inactive ones
    original_sample = sample.clone()
    
    step_index = [self.index_for_timestep[i][t.item()] for i, t in enumerate(timestep)]
    prev_step_index = [step+1 for step in step_index]
    
    # Extract sigmas for current and next steps (batched)
    # Each batch element gets its sigma from its corresponding step
    sigma = torch.tensor([self.sigmas[idx][i] for i, idx in enumerate(step_index)], 
                        device=sample.device, dtype=sample.dtype)
    sigma_prev = torch.tensor([self.sigmas[prev_idx][i] for i, prev_idx in enumerate(prev_step_index)],
                             device=sample.device, dtype=sample.dtype)
    
    # Handle inactive samples: replace sigma=0 with safe values to prevent NaN
    if active_mask is not None:
        # For inactive samples, use sigma=0.01 (a safe small value) to prevent division by zero
        safe_sigma = torch.where(active_mask, sigma, torch.tensor(0.01, device=sigma.device, dtype=sigma.dtype))
        safe_sigma_prev = torch.where(active_mask, sigma_prev, torch.tensor(0.01, device=sigma_prev.device, dtype=sigma_prev.dtype))
    else:
        safe_sigma = sigma
        safe_sigma_prev = sigma_prev
    
    # Reshape for broadcasting with sample dimensions
    sigma = safe_sigma.view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = safe_sigma_prev.view(-1, *([1] * (len(sample.shape) - 1)))
    
    # sigma_max is just the second sigma tensor, reshaped for broadcasting
    sigma_max = self.sigmas[1].view(-1, *([1] * (len(sample.shape) - 1)))
        
    dt = sigma_prev - sigma
    
    # Debug: Always show sigma values for monitoring
    #print(f"sigma: {sigma.flatten()}")
    #print(f"sigma_prev: {sigma_prev.flatten()}")
    
    # Debug: Check dt values for potential issues
    if torch.isnan(dt).any() or torch.isinf(dt).any():
        print(f"NaN/Inf detected in dt: {dt}")
    # Note: dt should be negative for denoising (sigma_prev - sigma < 0), but when both are 0, dt=0 is expected
    if (dt > 0).any():  # Only warn for strictly positive dt (excluding zero)
        positive_dt_mask = dt > 0
        print(f"Warning: Positive dt detected (should be negative for denoising): {dt[positive_dt_mask]}")

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
    
    # Debug: Check intermediate calculations for std_dev_t
    denominator = (1 - torch.where(sigma == 1, sigma_max, sigma))
    if (denominator <= 0).any():
        print(f"Warning: Non-positive denominator in std_dev_t calculation: {denominator.flatten()}")
        print(f"sigma: {sigma.flatten()}")
        print(f"sigma_max: {sigma_max.flatten()}")
    
    ratio_term = sigma / denominator
    if torch.isnan(ratio_term).any() or torch.isinf(ratio_term).any() or (ratio_term < 0).any():
        print(f"Warning: Invalid ratio_term for sqrt: {ratio_term.flatten()}")
    
    # Fix: Handle the case where sigma=0 or very small, which makes std_dev_t=0
    # When sigma is 0 or very small, use a minimum threshold for numerical stability
    min_std_dev = 1e-6  # Minimum standard deviation to prevent division by zero
    std_dev_t = torch.clamp(std_dev_t, min=min_std_dev)
    
    # Fix: Handle division by zero when sigma=0 in the prev_sample_mean calculation
    # When sigma=0, the terms involving division by sigma should be handled carefully
    safe_sigma = torch.clamp(sigma, min=1e-8)  # Prevent division by zero
    prev_sample_mean = sample*(1+std_dev_t**2/(2*safe_sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*safe_sigma))*dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    # For inactive samples, preserve the original sample
    if active_mask is not None:
        prev_sample = torch.where(active_mask.view(-1, *([1] * (len(sample.shape) - 1))), 
                                 prev_sample, 
                                 original_sample)
        # Also preserve original sample for prev_sample_mean to prevent NaN in log_prob calculation
        prev_sample_mean = torch.where(active_mask.view(-1, *([1] * (len(sample.shape) - 1))), 
                                      prev_sample_mean, 
                                      original_sample)

    # Debug: Check log probability calculation components for critical issues
    sqrt_neg_dt = torch.sqrt(-1*dt)
    if torch.isnan(sqrt_neg_dt).any() or torch.isinf(sqrt_neg_dt).any():
        print(f"NaN/Inf detected in sqrt(-dt): {sqrt_neg_dt.flatten()}")
        # Temporary fix: clamp dt to ensure -dt is positive
        dt = torch.clamp(dt, max=-1e-8)  # Ensure dt is sufficiently negative
        sqrt_neg_dt = torch.sqrt(-1*dt)
    
    variance_term = std_dev_t * sqrt_neg_dt
    if (variance_term <= 0).any():
        print(f"Warning: Non-positive variance_term for log: {variance_term.flatten()}")
        # Temporary fix: clamp to positive values
        variance_term = torch.clamp(variance_term, min=1e-8)
    
    # Compute log_prob with proper numerical stability
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (variance_term**2))
        - torch.log(variance_term)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # Set log_prob to 0 for inactive samples
    if active_mask is not None:
        log_prob = torch.where(active_mask, log_prob, torch.zeros_like(log_prob))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t