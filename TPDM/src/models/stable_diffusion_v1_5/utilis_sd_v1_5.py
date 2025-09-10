from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.outputs import BaseOutput


@dataclass
class CustomizedDiffusionModelOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    policy_inputs: any
    policy_outputs: any
    images: Union[
        List[PIL.Image.Image], np.ndarray, List[List[PIL.Image.Image]]
    ]  # A list of all the generated images for each sample and each timestep, all samples has the sample len, but maybe some are shorter than others, see prob_masks
    last_valid_indices: Optional[List[int]]  # indicate the last valid index for each sample
    alphas: Union[List[List[float]], torch.Tensor]
    betas: Union[List[List[float]], torch.Tensor]
    times: Union[List[List[float]], torch.Tensor]
    logprobs: Union[
        List[List[float]], torch.Tensor
    ]  # A list of log probabilities for each sample and each timestep, all samples has the sample len, but maybe some are shorter than others, see prob_masks
    prob_masks: Union[
        List[List[float]], torch.Tensor
    ]  # 1 if the sample is still being generated, 0 if it has finished but padding according to the batch size


class CustomizedDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None):
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return

        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(device=device, dtype=torch.int64)  # Ignore the last 0

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def customized_convert_model_output(self, model_output, sample, sigma=None):
        """
        TODO: considering other solver
        """

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if len(sigma.shape) != 0:
            sigma = sigma.view(-1, 1, 1, 1)
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                if sigma is None:
                    sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t

            return x0_pred
        else:
            ValueError("Only support dpmsolver++ and sde-dpmsolver++")

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        sigma_t: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")

        if sigma_t is None or sigma_s is None:
            sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        sigma_s = sigma_s.view(-1, 1, 1, 1)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        sigma_t: Optional[torch.Tensor] = None,
        sigma_s0: Optional[torch.Tensor] = None,
        sigma_s1: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")

        if sigma_t is None or sigma_s0 is None or sigma_s1 is None:
            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )

        sigma_t = sigma_t.view(-1, 1, 1, 1)
        sigma_s0 = sigma_s0.view(-1, 1, 1, 1)
        sigma_s1 = sigma_s1.view(-1, 1, 1, 1)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )

        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sigma_t: Optional[torch.Tensor] = None,
        sigma_s0: Optional[torch.Tensor] = None,
        sigma_s1: Optional[torch.Tensor] = None,
    ):
        """
        only considering dpmsolver++
        """
        if self.step_index is None:
            if len(timestep.shape) == 0:
                self._init_step_index(timestep)
            else:
                self._init_step_index(timestep[0])
        model_output = self.customized_convert_model_output(model_output, sample=sample, sigma=sigma_s0)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        noise = None
        final_timestep_is_zero_mask = timestep == 0
        final_timestep_is_nozero_mask = timestep != 0
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or final_timestep_is_zero_mask.all():
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise, sigma_t=sigma_t, sigma_s=sigma_s0)
        elif (
            (self.config.solver_order == 2
            or self.lower_order_nums < 2
            or lower_order_second)
            and (not final_timestep_is_zero_mask.any())
        ):
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise, sigma_t=sigma_t, sigma_s0=sigma_s0, sigma_s1=sigma_s1)
        elif final_timestep_is_zero_mask.any():
            # if the sample with final sigma is zero use the first order update, not zero use the second order update
            # if sigma_t == sigma_s0, con
            zero_prev_sample = self.dpm_solver_first_order_update(
                model_output[final_timestep_is_zero_mask],
                sample=sample[final_timestep_is_zero_mask],
                noise=noise,
                sigma_t=sigma_t[final_timestep_is_zero_mask],
                sigma_s=sigma_s0[final_timestep_is_zero_mask],
            )
            non_zero_sub_model_outputs = []
            non_zero_sub_model_outputs.append(self.model_outputs[0][final_timestep_is_nozero_mask])
            non_zero_sub_model_outputs.append(self.model_outputs[1][final_timestep_is_nozero_mask])
            none_zreo_prev_sample = self.multistep_dpm_solver_second_order_update(non_zero_sub_model_outputs, sample=sample[final_timestep_is_nozero_mask], noise=noise, sigma_t=sigma_t[final_timestep_is_nozero_mask], sigma_s0=sigma_s0[final_timestep_is_nozero_mask], sigma_s1=sigma_s1[final_timestep_is_nozero_mask])
            prev_sample = torch.zeros_like(model_output)
            prev_sample[final_timestep_is_zero_mask] = zero_prev_sample
            prev_sample[final_timestep_is_nozero_mask] = none_zreo_prev_sample
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1
        return (prev_sample,)
