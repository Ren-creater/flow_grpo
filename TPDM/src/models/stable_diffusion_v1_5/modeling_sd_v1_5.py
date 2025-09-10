# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import ConfigMixin, ModelMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor

from .unet_sd_v1_5 import CustomizedUNet2DConditionModel
from .utilis_sd_v1_5 import CustomizedDiffusionModelOutput, CustomizedDPMSolverMultistepScheduler


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CustomAdaGroupNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, input_dim: int, embedding_dim: int, norm_type="group_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(input_dim, 2 * embedding_dim, bias=bias)
        if norm_type == "group_norm":
            self.norm = nn.GroupNorm(1, embedding_dim, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, :, None, None]) + shift_msa[:, :, None, None]
        return x

class TimePredictor(nn.Module):
    def __init__(
        self,
        conv_out_channels,
        in_channels=1536 * 2,
        projection_dim=2,
        init_alpha=1.5,
        init_beta=-0.7,
    ):
        super(TimePredictor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            conv_out_channels,
            conv_out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=2,
        )

        self.fc1 = nn.Linear(conv_out_channels, 128)
        self.fc2 = nn.Linear(128, projection_dim)

        self.norm1 = CustomAdaGroupNormZeroSingle(in_channels // 2, conv_out_channels)
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self._init_weights()

    def forward(self, x, temb):
        # 输入张量形状: (bs, 320, 64, 64), (bs, 320)

        # (bs, 320, 64, 64) -> (bs, conv_out_channels, 64, 64)
        x = self.conv1(x)
        x = self.norm1(x, temb)
        # TODO: add timestep information by a layernorm with t-related weight & bias, similar to AdaLN
        x = F.silu(x)
        x = self.conv2(x)
        # (bs, conv_out_channels, 32, 32) -> (bs, conv_out_channels, 1, 1) -> (bs, conv_out_channels)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        x = F.silu(self.fc1(x))  # (bs, 128)
        x = self.fc2(x)  # (bs, 2)
        return torch.exp(x) + 1.0

    def _init_weights(self):
        # init it by std is 0.02 and mean is 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None and isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias[0], self.init_alpha)
        nn.init.constant_(self.fc2.bias[1], self.init_beta)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SD15PredictNextTimeStepModel(ModelMixin, ConfigMixin):

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _supports_gradient_checkpointing = False
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D"]

    def __init__(
        self,
        pretrained_model_name_or_path = None,
        do_train=False,
        init_alpha=1.5,
        init_beta=-0.7,
        freeze_layers = [],
        gamma=0.97,
    ):
        super().__init__()

        self.tokenizer = None
        self.text_encoder = None
        if not do_train:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

        self.unet = CustomizedUNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

        self.scheduler = CustomizedDPMSolverMultistepScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.time_predictor = TimePredictor(
            conv_out_channels=128,
            in_channels=320 * 2,  # block_out_channels[0] * 2
            projection_dim=2,
            init_alpha=init_alpha,
            init_beta=init_beta,
        ).to(dtype=self.unet.dtype)

        self._is_unet_config_sample_size_int = isinstance(self.unet.config.sample_size, int)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        for layer in freeze_layers:
            if hasattr(self, layer):
                getattr(self, layer).requires_grad_(False)

        self.epsilon = 1e-3
        self.min_time = 10
        self.gamma = gamma
        self.all_sigmas = torch.from_numpy(np.array(((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod) ** 0.5))

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        predict: bool = False,
        fixed_policy_inputs: Optional[Dict[str, List[torch.Tensor]]] = None,
        fixed_policy_outputs: Optional[Dict[str, List[torch.Tensor]]] = None,
        **kwargs,
    ):


        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.unet.device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        self.all_sigmas = self.all_sigmas.to(device)
        t = torch.ones(batch_size, dtype=torch.int64, device=device) * 999
        history_latents = [[] for _ in range(batch_size)]
        times = [[] for _ in range(batch_size)]
        logprobs = [[] for _ in range(batch_size)]
        prob_masks = [[] for _ in range(batch_size)]
        alphas = [[] for _ in range(batch_size)]
        betas = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            times[i].append(t[i])

        policy_inputs = {
            "latents": [[] for _ in range(batch_size)],
            "t": [[] for _ in range(batch_size)],
        }
        policy_outputs = {"ratio": [[] for _ in range(batch_size)]}

        if fixed_policy_inputs is not None:
            for key in fixed_policy_inputs.keys():
                num_dim = len(fixed_policy_inputs[key].shape)
                if num_dim == 2:
                    fixed_policy_inputs[key] = fixed_policy_inputs[key].permute(1, 0)
                elif num_dim == 5:
                    fixed_policy_inputs[key] = fixed_policy_inputs[key].permute(1, 0, 2, 3, 4)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self._num_timesteps = len(timesteps)

        for step in range(num_inference_steps):

            # record policy inputs
            if fixed_policy_inputs is not None:
                latents = fixed_policy_inputs["latents"][step]
                t = fixed_policy_inputs["t"][step]
            else:
                for i in range(batch_size):
                    policy_inputs["latents"][i].append(latents[i].detach())
                    policy_inputs["t"][i].append(t[i].detach())

            # expand the latents if we are doing classifier free guidance
            latents = latents.to(dtype=self.unet.dtype)
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            t = t.repeat(2) if self.do_classifier_free_guidance else t

            # predict the noise residual
            # (bs, 4, 64, 64), (bs, 320), (bs, 320, 64, 64), (bs, 320, 64, 64)
            noise_pred, temb, hidden_states_1, hidden_states_2 = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=self.cross_attention_kwargs,
                return_dict=False,
            )

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                (noise_pred_text - noise_pred_uncond)
                temb_uncond, temb_text = temb.chunk(2)
                temb = temb_uncond + guidance_scale * (temb_text - temb_uncond)  # guidance_scale * (temb_text - temb_uncond) == 0
                hidden_states_1_uncond, hidden_states_1_text = hidden_states_1.chunk(2)
                hidden_states_1 = hidden_states_1_uncond + guidance_scale * (
                    hidden_states_1_text - hidden_states_1_uncond
                )
                hidden_states_2_uncond, hidden_states_2_text = hidden_states_2.chunk(2)
                hidden_states_2 = hidden_states_2_uncond + guidance_scale * (
                    hidden_states_2_text - hidden_states_2_uncond
                )
                t = t[:batch_size]

            hidden_states_combined = torch.cat([hidden_states_1, hidden_states_2], dim=1)
            time_preds = self.time_predictor(hidden_states_combined, temb)
            t_next = torch.zeros_like(t)


            for i, (param1, param2) in enumerate(time_preds):
                alpha = param1
                beta = param2
                beta_dist = torch.distributions.Beta(alpha, beta)

                if predict:
                    ratio = beta_dist.mode
                else:
                    ratio = beta_dist.sample()
                ratio = ratio.clamp(self.epsilon, 1 - self.epsilon)

                if fixed_policy_outputs is not None:
                    ratio = fixed_policy_outputs["ratio"][i][step]
                else:
                    policy_outputs["ratio"][i].append(ratio)

                t_next[i] = t[i] * ratio

                prob = beta_dist.log_prob(ratio)
                logprobs[i].append(prob)

                if t[i] < self.min_time:
                    prob_masks[i].append(torch.tensor(1))
                    t_next[i] = torch.tensor(0).to(device=ratio.device)
                else:
                    prob_masks[i].append(torch.tensor(0))

                times[i].append(t_next[i])
                alphas[i].append(alpha)
                betas[i].append(beta)

            timestep_t = t_next.to(torch.int64)
            timestep_s0 = t.to(torch.int64)
            timestep_s1 = torch.zeros_like(t)
            if step > 0:
                for i in range(batch_size):
                    timestep_s1[i] = times[i][-3].to(torch.int64)

            if step == 0:
                sigma_t = self.all_sigmas[timestep_t]
                sigma_s0 = self.all_sigmas[timestep_s0]
                sigma_s1 = None
            elif step == num_inference_steps - 1:
                sigma_t = torch.tensor([0]*batch_size, device=timesteps.device, dtype=torch.int64)
                sigma_s0 = self.all_sigmas[timestep_s0]
                sigma_s1 = self.all_sigmas[timestep_s1]
            else:
                sigma_t = self.all_sigmas[timestep_t]
                sigma_s0 = self.all_sigmas[timestep_s0]
                sigma_s1 = self.all_sigmas[timestep_s1]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t_next,
                latents,
                **extra_step_kwargs,
                sigma_t=sigma_t,
                sigma_s0=sigma_s0,
                sigma_s1=sigma_s1,
                return_dict=False,
            )[0]

            for i in range(len(history_latents)):
                # record the latents
                if len(history_latents[i]) == 0:
                    history_latents[i] = latents[i].detach().unsqueeze(0)
                else:
                    history_latents[i] = torch.cat(
                        [history_latents[i], latents[i].detach().unsqueeze(0)], dim=0
                    )

            if (t_next == 0).all():
                break

            t = t_next

        INVALID_LOGPROB = 1.0
        times = torch.stack([torch.stack(item) for item in times])
        logprobs = torch.stack([torch.stack(item) for item in logprobs])
        prob_masks = torch.stack([torch.stack(item) for item in prob_masks]).bool().to(logprobs.device)
        alphas = torch.stack([torch.stack(item) for item in alphas])
        betas = torch.stack([torch.stack(item) for item in betas])
        logprobs = torch.masked_fill(logprobs, prob_masks, INVALID_LOGPROB)

        # policy inputs and outputs
        if fixed_policy_inputs is None and fixed_policy_outputs is None:
            for key in policy_inputs.keys():
                policy_inputs[key] = torch.stack([torch.stack(item) for item in policy_inputs[key]])
            for key in policy_outputs.keys():
                policy_outputs[key] = torch.stack([torch.stack(item) for item in policy_outputs[key]])

        images = []
        last_valid_indices = []
        for i in range(prob_masks.shape[0]):
            last_valid_index = torch.where(~prob_masks[i])[0][-1]
            last_valid_indices.append(last_valid_index)

        for i, latents in enumerate(history_latents):
            last_valid_index = last_valid_indices[i]
            latents = latents[last_valid_index]
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(
                latents.unsqueeze(0) / self.vae.config.scaling_factor, return_dict=False, generator=generator
            )[0]
            do_denormalize = [True] * image.shape[0]
            images.append(
                self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)[0]
            )


        if not return_dict:
            return (images, policy_inputs, policy_outputs, last_valid_indices, alphas, betas, times, logprobs, prob_masks)

        return CustomizedDiffusionModelOutput(
            images=images,
            policy_inputs=policy_inputs,
            policy_outputs=policy_outputs,
            last_valid_indices=last_valid_indices,
            alphas=alphas,
            betas=betas,
            times=times,
            logprobs=logprobs,
            prob_masks=prob_masks,
        )

    def rloo_repeat(self, data, rloo_k=2):
        """make the data repeat rloo_k times
        Args:
            data: dict of data
            rloo_k: int
        Returns:
            data: dict of data that is repeated rloo_k times
        """
        data["prompt"] = data["prompt"] * rloo_k
        return data

    def sample(self, inputs):
        """Generate model outputs step by step for inputs
        Args:
            inputs: dict of inputs
        Returns:
            outputs: dict of final outputs after sampling
        """
        outputs = self.forward(**inputs)
        return outputs

    def reward(self, inputs, outputs, reward_model, prompt, return_last_reward=False):
        """Given a batch of model inputs and outputs, provide the rewards of the outputs, using the final image in outputs
        Args:
            inputs: dict of inputs
            outputs: dict of outputs
            reward_model: reward model
            return_last_reward: whether to return the last reward
        Returns:
            rewards: tensor of rewards (bs, )
        """
        gamma = self.gamma
        prompts = inputs.get("prompt", None) if prompt is None else prompt
        images = outputs.get("images", None)
        prob_masks = outputs.get("prob_masks", None)
        last_valid_indices = outputs.get("last_valid_indices", [])
        rewards = []
        last_image_rewards = []
        if prompts is None or images is None:
            raise ValueError("prompt and images must be provided")
        elif len(prompts) != len(images):
            raise ValueError("prompt and images must have the same length")
        for i, (prompt, image, prob_mask) in enumerate(zip(prompts, images, prob_masks)):
            # use last image where prob_mask is false to calculate reward, and use gamma to discount the reward
            if last_valid_indices == []:
                last_image_idx = torch.where(~prob_mask.bool())[-1][-1].item()
                last_image = image[last_image_idx]
            else:
                last_image_idx = last_valid_indices[i].item()
                last_image = image
            last_image_reward = reward_model.score(prompt, last_image)
            last_image_rewards.append(last_image_reward)
            reward = 0
            for i in range(last_image_idx + 1):
                reward += last_image_reward * (gamma ** (last_image_idx - i))
            reward = reward / (last_image_idx + 1)
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        last_image_rewards = torch.tensor(last_image_rewards)
        if return_last_reward:
            return rewards, last_image_rewards
        else:
            return rewards

    def logprobs(self, inputs, outputs):
        """Given a batch of model inputs and outputs, provide the logprobs of the outputs, using the actions in outputs
        Args:
            outputs: dict of outputs
        Returns:
            logprobs: tensor of logprobs (bs, seq_len)
            prob_masks: tensor of masks for the logprobs (bs, seq_len)
        """
        outputs = self.forward(
            **inputs,
            fixed_policy_inputs=outputs["policy_inputs"],
            fixed_policy_outputs=outputs["policy_outputs"],
        )
        return outputs

    def kl_divergence(self, inputs, outputs):
        """kl_divergence is not needed"""
        batch_size = len(outputs["images"])
        kl_div = torch.tensor([0.0] * batch_size)
        return kl_div

    def subset_inputs(self, inputs, micro_batch_inds):
        subset_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                subset_inputs[key] = value[micro_batch_inds]
            elif isinstance(value, list):
                subset_inputs[key] = [value[i] for i in micro_batch_inds]
            elif isinstance(value, float) or isinstance(value, int) or value is None:
                subset_inputs[key] = value
            else:
                raise ValueError(f"Unsupported input type: {type(value)}")
        return subset_inputs

    def subset_outputs(self, outputs, micro_batch_inds):
        subset_outputs = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                subset_outputs[key] = value[micro_batch_inds]
            elif isinstance(value, list):
                subset_outputs[key] = [value[i] for i in micro_batch_inds]
            elif isinstance(value, dict):
                sub_dict = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        sub_dict[k] = v[micro_batch_inds]
                    else:
                        ValueError(f"Unsupported output type: {type(v)}")
                subset_outputs[key] = sub_dict
            else:
                raise ValueError(f"Unsupported output type: {type(value)}")
        return subset_outputs

if __name__ == "__main__":
    model_id = "models/runwayml/stable-diffusion-v1-5"
    pipe = SD15PredictNextTimeStepModel(model_id).to("cuda")
    pipe = pipe.cuda()
    sampling_schedule = [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0]
    with torch.no_grad():
        output = pipe(
            prompt="a cat in the bathroom",
            num_inference_steps=10,
            # timesteps=sampling_schedule,
            generator=torch.Generator().manual_seed(42),
        )
        images = output["images"]
    images[0].save("test.png")
