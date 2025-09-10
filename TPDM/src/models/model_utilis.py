from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.outputs import BaseOutput


@dataclass
class CustomFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

@dataclass
class CustomDiffusionModelOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    init_noise_latents: torch.Tensor  #
    hidden_states_combineds: torch.Tensor
    tembs: torch.Tensor
    images: Union[List[PIL.Image.Image], np.ndarray, List[List[PIL.Image.Image]]] # A list of all the generated images for each sample and each timestep, all samples has the sample len, but maybe some are shorter than others, see prob_masks
    last_valid_indices: Optional[List[int]]                                       # indicate the last valid index for each sample
    alphas: Union[List[List[float]], torch.Tensor]
    betas: Union[List[List[float]], torch.Tensor]
    sigmas: Union[List[List[float]], torch.Tensor]
    logprobs: Union[List[List[float]], torch.Tensor]                              # A list of log probabilities for each sample and each timestep, all samples has the sample len, but maybe some are shorter than others, see prob_masks
    prob_masks: Union[List[List[float]], torch.Tensor]                            # 1 if the sample is still being generated, 0 if it has finished but padding according to the batch size


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_step(
        self,
        model_output: torch.FloatTensor,
        sigma_next: float,
        sigma: float,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[CustomFlowMatchEulerDiscreteSchedulerOutput, Tuple]:

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        delta_sigma = sigma_next - sigma
        delta_sigma = delta_sigma.view(-1, 1, 1, 1)

        prev_sample = sample + delta_sigma * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return CustomFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
