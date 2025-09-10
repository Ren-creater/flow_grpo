import logging
from typing import List, Optional, Union
from dataclasses import dataclass

import pyrootutils
import torch
import torch.distributed.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from transformers.trainer import TrainingArguments

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.normalization import AdaLayerNormZeroSingle
from diffusers.utils.torch_utils import randn_tensor


pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from src.models.model_utilis import CustomDiffusionModelOutput, CustomFlowMatchEulerDiscreteScheduler
from src.models.reference_distributions import get_ref_beta
from src.models.stable_diffusion_3.transformer_sd3 import CustomSD3Transformer2DModel


logger = logging.getLogger(__name__)

from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

# def reshape_hidden_states_to_2d(
#     hidden_states: torch.Tensor,
#     height: int = 64,
#     width: int = 64,
#     patch_size: int = 2,
# ) -> torch.Tensor:
#     """
#     Reshape the hidden states to have a 2D spatial structure.
#     """
#     hidden_states = hidden_states.reshape(
#         shape=(
#             hidden_states.shape[0],
#             height // patch_size,
#             width // patch_size,
#             patch_size,
#             patch_size,
#             hidden_states.shape[-1],
#         )
#     )
#     hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
#     hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], hidden_states.shape[1], height, width))
#     return hidden_states

def reshape_hidden_states_to_2d(hidden_states, height=64, width=64, patch_size=2):
    B, L, C = hidden_states.shape  # e.g. (16, 1024, 1536)

    Hpatch, Wpatch = height // patch_size, width // patch_size
    assert L == Hpatch * Wpatch, f"Expected {Hpatch*Wpatch} tokens, got {L}"

    # Do NOT divide channels by patch_size**2
    # Keep full C (1536), just expand spatially
    x = hidden_states.reshape(B, Hpatch, Wpatch, C)
    x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, Hpatch, Wpatch)
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="nearest")
    return x

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

    def forward(
        self,
        x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, :, None, None]) + shift_msa[:, :, None, None]
        return x

class TimePredictor(nn.Module):
    def __init__(self, conv_out_channels, in_channels=1536 * 2, projection_dim=2, init_alpha=1.5, init_beta=0.5):
        super(TimePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=(3, 3), padding=1, stride=2)

        self.fc1 = nn.Linear(conv_out_channels, 128)
        self.fc2 = nn.Linear(128, projection_dim)

        self.norm1 = CustomAdaGroupNormZeroSingle(in_channels // 2, conv_out_channels)
        self.epsilon = 1.0
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self._init_weights()

    def forward(self, x, temb):
        # 输入张量形状: (bs, 3072, 64, 64), (bs, 1536)

        # (bs, 3072, 64, 64) -> (bs, conv_out_channels, 64, 64)
        x = self.conv1(x)
        x = self.norm1(x, temb)
        # TODO: add timestep information by a layernorm with t-related weight & bias, similar to AdaLN
        x = F.silu(x)
        x = self.conv2(x)
        # (bs, conv_out_channels, 64, 64) -> (bs, conv_out_channels, 1, 1) -> (bs, conv_out_channels)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        x = F.silu(self.fc1(x))  # (bs, 128)
        x = self.fc2(x)  # (bs, 2)
        return torch.exp(x) + self.epsilon

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


@dataclass
class TimePredictorConfig:
    """Configuration class for TimePredictor models."""
    # Vision Transformer specific parameters
    num_layers: int = 6
    num_heads: int = 8
    hidden_size: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Input/Output parameters
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3072  # Combined hidden states channels
    text_embed_dim: int = 1536  # Text embedding dimension
    timestep_embed_dim: int = 1536  # Timestep embedding dimension
    projection_dim: int = 2
    
    # Initialization parameters
    init_alpha: float = 1.5
    init_beta: float = 0.5
    epsilon: float = 1.0
    
    # Model architecture choice
    use_text_conditioning: bool = True
    use_timestep_conditioning: bool = True
    cross_attention: bool = True
    
    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (self.image_size // self.patch_size) ** 2


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, context=None, attention_mask=None):
        B, N, C = x.shape
        if context is None:
            context = x
        
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: TimePredictorConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            config.hidden_size, 
            config.num_heads, 
            config.attention_dropout
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        
        # Cross-attention (for text conditioning)
        if config.use_text_conditioning and config.cross_attention:
            self.cross_attn = MultiHeadAttention(
                config.hidden_size,
                config.num_heads,
                config.attention_dropout
            )
            self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Timestep conditioning via AdaLN (Adaptive Layer Normalization)
        if config.use_timestep_conditioning:
            self.timestep_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.timestep_embed_dim, 2 * config.hidden_size)
            )
            self.norm_timestep = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        
        # MLP
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.norm3 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, text_context=None, timestep_embed=None, attention_mask=None):
        # Self-attention with optional timestep conditioning
        if hasattr(self, 'timestep_proj') and timestep_embed is not None:
            # AdaLN: modulate normalization with timestep
            norm_x = self.norm_timestep(x)
            timestep_params = self.timestep_proj(timestep_embed)  # (B, 2*hidden_size)
            shift, scale = timestep_params.chunk(2, dim=-1)  # (B, hidden_size) each
            shift = shift.unsqueeze(1)  # (B, 1, hidden_size)
            scale = scale.unsqueeze(1)  # (B, 1, hidden_size)
            norm_x = norm_x * (1 + scale) + shift
            x = x + self.self_attn(norm_x, attention_mask=attention_mask)
        else:
            # Standard layer normalization
            x = x + self.self_attn(self.norm1(x), attention_mask=attention_mask)
        
        # Cross-attention with text
        if hasattr(self, 'cross_attn') and text_context is not None:
            x = x + self.cross_attn(self.norm2(x), context=text_context)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class ViTTimePredictor(nn.Module):
    """Vision Transformer-based TimePredictor with text conditioning support."""
    
    def __init__(self, config: TimePredictorConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels, 
            config.hidden_size, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.hidden_size)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Text projection (if using text conditioning)
        if config.use_text_conditioning:
            self.text_proj = nn.Linear(config.text_embed_dim, config.hidden_size)
            self.text_norm = nn.LayerNorm(config.hidden_size)
        
        # Timestep projection (if using timestep conditioning)
        if config.use_timestep_conditioning:
            self.timestep_proj = nn.Linear(config.timestep_embed_dim, config.hidden_size)
            self.timestep_norm = nn.LayerNorm(config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization and projection
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.projection_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.epsilon = config.epsilon
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0)
        
        # Initialize head with specific alpha/beta values
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias[0], self.config.init_alpha)
        nn.init.constant_(self.head.bias[1], self.config.init_beta)
        
        # Initialize text projection if exists
        if hasattr(self, 'text_proj'):
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.constant_(self.text_proj.bias, 0)
    
    def forward(self, hidden_states_combined, text_embeds=None, timestep_embed=None, attention_mask=None):
        """
        Args:
            hidden_states_combined: (B, C, H, W) - Combined visual features
            text_embeds: (B, seq_len, text_embed_dim) - Text embeddings
            timestep_embed: (B, timestep_embed_dim) - Timestep embeddings
            attention_mask: (B, seq_len) - Attention mask for text
        """
        B = hidden_states_combined.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = self.patch_embed(hidden_states_combined)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, hidden_size)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Process text embeddings if provided
        text_context = None
        if self.config.use_text_conditioning and text_embeds is not None:
            # Project text embeddings to hidden size
            text_context = self.text_proj(text_embeds)  # (B, seq_len, hidden_size)
            text_context = self.text_norm(text_context)
        
        # Process timestep embeddings if provided
        processed_timestep_embed = None
        if self.config.use_timestep_conditioning and timestep_embed is not None:
            # Project timestep embeddings to hidden size
            processed_timestep_embed = self.timestep_proj(timestep_embed)  # (B, hidden_size)
            processed_timestep_embed = self.timestep_norm(processed_timestep_embed)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, text_context=text_context, timestep_embed=processed_timestep_embed, attention_mask=attention_mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Use class token for prediction
        cls_output = x[:, 0]  # (B, hidden_size)
        
        # Project to alpha, beta
        output = self.head(cls_output)  # (B, 2)
        
        # Apply exponential and add epsilon for numerical stability
        return torch.exp(output) + self.epsilon


class HybridTimePredictor(nn.Module):
    """Hybrid model that can use either CNN or ViT architecture."""
    
    def __init__(self, config: TimePredictorConfig, use_vit: bool = True):
        super().__init__()
        self.config = config
        self.use_vit = use_vit
        
        if use_vit:
            self.predictor = ViTTimePredictor(config)
        else:
            # Fall back to original CNN-based TimePredictor
            self.predictor = TimePredictor(
                conv_out_channels=128,
                in_channels=config.in_channels,
                projection_dim=config.projection_dim,
                init_alpha=config.init_alpha,
                init_beta=config.init_beta
            )
    
    def forward(self, hidden_states_combined, text_embeds=None, timestep_embed=None, attention_mask=None):
        if self.use_vit:
            return self.predictor(hidden_states_combined, text_embeds, timestep_embed, attention_mask)
        else:
            # For CNN version, we use timestep embeddings as temb (temporal embedding)
            # This maintains compatibility with the original CNN TimePredictor
            if timestep_embed is not None:
                temb = timestep_embed
            elif text_embeds is not None:
                temb = text_embeds.mean(dim=1)  # Pool sequence dimension as fallback
            else:
                # Create dummy temb with correct shape
                B = hidden_states_combined.shape[0]
                temb = torch.zeros(B, self.config.timestep_embed_dim, 
                                 device=hidden_states_combined.device,
                                 dtype=hidden_states_combined.dtype)
            return self.predictor(hidden_states_combined, temb)

def init_time_predictor(
    self,
    pretrained_model_name_or_path,
    min_sigma=0.001,
    init_alpha=1.5,
    init_beta=0.5,
    relative=True,
    prediction_type="alpha_beta",
    use_vit_predictor=False,
    time_predictor_config=None,
):

    # Initialize TimePredictor (CNN or ViT based)
    if use_vit_predictor:
        if time_predictor_config is None:
            time_predictor_config = TimePredictorConfig(
                in_channels=self.transformer.config.caption_projection_dim * 2,
                text_embed_dim=self.transformer.config.caption_projection_dim,
                init_alpha=init_alpha,
                init_beta=init_beta,
            )
        self.time_predictor = HybridTimePredictor(time_predictor_config, use_vit=True)
    else:
        self.time_predictor = TimePredictor(
            conv_out_channels=128,
            in_channels=self.transformer.config.caption_projection_dim * 2,
            projection_dim=2,
            init_alpha=init_alpha,
            init_beta=init_beta,
        )
    
    self.time_predictor = self.time_predictor.to(dtype=self.vae.dtype)
    self.use_vit_predictor = use_vit_predictor

    self.min_sigma = min_sigma
    self.relative = relative
    self.epsilon = 1e-3
    self.prediction_type = prediction_type

    load_time_predictor(self, pretrained_model_name_or_path)


def load_time_predictor(pipeline, ckpt_path):
    from safetensors.torch import load_file
    state_dict = load_file(ckpt_path)

    # Check if keys already start with "time_predictor."
    if all(k.startswith("time_predictor.") for k in state_dict.keys()):
        # Strip the prefix so it matches the submodule
        time_predictor_state = {
            k.replace("time_predictor.", "", 1): v for k, v in state_dict.items()
        }
    else:
        # Assume it is already the submodule dict
        time_predictor_state = state_dict

    # Load into the submodule
    missing, unexpected = pipeline.time_predictor.load_state_dict(
        time_predictor_state, strict=False
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

class SD3PredictNextTimeStepModel(nn.Module, SD3LoraLoaderMixin):
    def __init__(
        self,
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        min_sigma=0.001,
        init_alpha=1.5,
        init_beta=0.5,
        pre_process=False,
        relative=True,
        prediction_type="alpha_beta",
        use_vit_predictor=False,
        time_predictor_config=None,
    ):
        super(SD3PredictNextTimeStepModel, self).__init__()

        # initialize the models
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype
        )
        self.transformer = CustomSD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )

        # Initialize TimePredictor (CNN or ViT based)
        if use_vit_predictor:
            if time_predictor_config is None:
                time_predictor_config = TimePredictorConfig(
                    in_channels=self.transformer.config.caption_projection_dim * 2,
                    text_embed_dim=self.transformer.config.caption_projection_dim,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                )
            self.time_predictor = HybridTimePredictor(time_predictor_config, use_vit=True)
        else:
            self.time_predictor = TimePredictor(
                conv_out_channels=128,
                in_channels=self.transformer.config.caption_projection_dim * 2,
                projection_dim=2,
                init_alpha=init_alpha,
                init_beta=init_beta,
            )
        
        self.time_predictor = self.time_predictor.to(dtype=self.vae.dtype)
        self.use_vit_predictor = use_vit_predictor
        self.scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        if not pre_process:
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype
            ).eval()
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch_dtype
            ).eval()
            self.text_encoder_3 = (
                T5EncoderModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=torch_dtype
                )
                .to(dtype=self.vae.dtype)
                .eval()
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
            self.tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")

        self.pre_process = pre_process
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 77
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

        self.min_sigma = min_sigma
        self.relative = relative
        self.epsilon = 1e-3
        self.prediction_type = prediction_type

        self.requires_grad_(False)
        self.eval()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def skip_guidance_layers(self):
        return self._skip_guidance_layers

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def _execution_device(self):
        return self.vae.device
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.vae.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_3: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        device = device or self.vae.device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (
                    0,
                    t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
                ),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: int = 1,
        max_inference_steps: int = 28,
        guidance_scale: Union[float, None] = 7.0,
        generator: Union[torch.Generator, List[torch.Generator]] = None,
        latents: Optional[torch.FloatTensor] = None,
        fix_sigmas: Optional[torch.FloatTensor] = None,
        return_full_process_images: bool = False,
        predict: bool = False,
    ) -> CustomDiffusionModelOutput:
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.vae.device,
                )
            )
        else:
            prompt_embeds = prompt_embeds.to(self.vae.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.vae.device)
            negative_prompt_embeds = (
                negative_prompt_embeds.to(self.vae.device) if negative_prompt_embeds is not None else None
            )
            negative_pooled_prompt_embeds = (
                negative_pooled_prompt_embeds.to(self.vae.device)
                if negative_pooled_prompt_embeds is not None
                else None
            )

        batch_size = prompt_embeds.shape[0]
        num_channels_latents = self.transformer.config.in_channels
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor
        device = self.vae.device

        if latents is None:
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
        init_noise_latents = latents.clone()

        if guidance_scale is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        sigma = torch.ones(batch_size, dtype=latents.dtype, device=device)
        history_latents = [[] for _ in range(batch_size)]
        sigmas = [[] for _ in range(batch_size)]
        logprobs = [[] for _ in range(batch_size)]
        prob_masks = [[] for _ in range(batch_size)]
        alphas = [[] for _ in range(batch_size)]
        betas = [[] for _ in range(batch_size)]
        now_step = 0

        hidden_states_combineds = []
        tembs = []
        # Denoising loop
        if fix_sigmas is not None:
            max_inference_steps = len(fix_sigmas[0])
        for step in range(max_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents.detach()] * 2) if guidance_scale else latents

            timestep = sigma.repeat(2) * 1000

            (noise_pred, temb, hidden_states_1, hidden_states_2) = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )

            if guidance_scale is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                temb_uncond, temb_text = temb.chunk(2)
                temb = temb_uncond + guidance_scale * (temb_text - temb_uncond)
                hidden_states_1_uncond, hidden_states_1_text = hidden_states_1.chunk(2)
                hidden_states_1 = hidden_states_1_uncond + guidance_scale * (
                    hidden_states_1_text - hidden_states_1_uncond
                )
                hidden_states_2_uncond, hidden_states_2_text = hidden_states_2.chunk(2)
                hidden_states_2 = hidden_states_2_uncond + guidance_scale * (
                    hidden_states_2_text - hidden_states_2_uncond
                )

            hidden_states_1 = reshape_hidden_states_to_2d(hidden_states_1)
            hidden_states_2 = reshape_hidden_states_to_2d(hidden_states_2)
            hidden_states_combined = torch.cat([hidden_states_1, hidden_states_2], dim=1)
            hidden_states_combineds.append(hidden_states_combined.cpu())
            tembs.append(temb)

            # Call time predictor with appropriate inputs
            if self.use_vit_predictor:
                # For ViT, extract text embeddings from prompt_embeds
                if guidance_scale is not None:
                    # During CFG, prompt_embeds contains [negative, positive] concatenated
                    # We want the positive part for text conditioning
                    text_embeds = prompt_embeds[batch_size:]  # Take the positive part
                    # Also extract negative part for timestep conditioning if needed
                    timestep_embeds = pooled_prompt_embeds[batch_size:]  # Use pooled embeds as timestep
                else:
                    text_embeds = prompt_embeds
                    timestep_embeds = pooled_prompt_embeds
                time_preds = self.time_predictor(hidden_states_combined, text_embeds, timestep_embeds)
            else:
                # For CNN, use pooled embeddings as temporal embedding (original behavior)
                time_preds = self.time_predictor(hidden_states_combined, temb)
            sigma_next = torch.zeros_like(sigma)
            for i, (param1, param2) in enumerate(time_preds):
                if self.prediction_type == "alpha_beta":
                    alpha, beta = param1, param2
                elif self.prediction_type == "mode_concentration":
                    alpha = param1 * (param2 - 2) + 1
                    beta = (1 - param1) * (param2 - 2) + 1
                beta_dist = torch.distributions.Beta(alpha, beta)

                if predict:
                    ratio = beta_dist.mode
                else:
                    ratio = beta_dist.sample()
                ratio = (
                    ratio.clamp(self.epsilon, 1 - self.epsilon)
                    if self.relative
                    else ratio.clamp(self.epsilon, sigma[i]).clamp(0, 1 - self.epsilon)
                )
                # TODO: Different map function attempts
                sigma_next[i] = sigma[i] * ratio if self.relative else sigma[i] - ratio
                sigmas[i].append(sigma_next[i])

                # if ratio is nan pdb
                prob = beta_dist.log_prob(ratio)
                logprobs[i].append(prob)
                if sigma[i] < self.min_sigma:
                    prob_masks[i].append(torch.tensor(1))
                    if predict:
                        sigma_next[i] = torch.tensor(0.0).to(sigma_next.device)
                else:
                    prob_masks[i].append(torch.tensor(0))

                alphas[i].append(alpha)
                betas[i].append(beta)

            latents = self.scheduler.custom_step(
                noise_pred,
                sigma_next=sigma_next,
                sigma=sigma,
                sample=latents,
                return_dict=False,
            )[0]

            for i in range(len(history_latents)):
                # record the latents
                if len(history_latents[i]) == 0:
                    history_latents[i] = latents[i].detach().unsqueeze(0)
                else:
                    history_latents[i] = torch.cat([history_latents[i], latents[i].detach().unsqueeze(0)], dim=0)

            # all sigma are lower than the threshold, we stop the chain
            if (sigma_next < self.min_sigma).all():
                break

            sigma = sigma_next
            now_step += 1

        # TODO: check
        INVALID_LOGPROB = 1.0
        sigmas = torch.stack([torch.stack(item) for item in sigmas])
        logprobs = torch.stack([torch.stack(item) for item in logprobs])
        prob_masks = torch.stack([torch.stack(item) for item in prob_masks]).bool().to(logprobs.device)
        alphas = torch.stack([torch.stack(item) for item in alphas])
        betas = torch.stack([torch.stack(item) for item in betas])
        logprobs = torch.masked_fill(logprobs, prob_masks, INVALID_LOGPROB)

        # (num_steps, batch_size, ...) -> (batch_size, num_steps, ...)
        hidden_states_combineds = torch.stack(hidden_states_combineds).permute(1, 0, 2, 3, 4)
        tembs = torch.stack(tembs).permute(1, 0, 2)

        images = []
        last_valid_indices = []
        if return_full_process_images:
            for latents in history_latents:
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                # chunk the latents into chunks of 8
                chunk_size = (len(latents) // 8) + 1
                chunk_latents = latents.chunk(chunk_size, dim=0)
                del latents
                image = None
                for latents in chunk_latents:
                    if image is None:
                        image = self.vae.decode(latents, return_dict=False)[0].detach()
                    else:
                        image = torch.cat([image, self.vae.decode(latents, return_dict=False)[0].detach()], dim=0)
                    torch.cuda.empty_cache()
                images.append(self.image_processor.postprocess(image, output_type="pil"))
        else:
            # only decode the last valid latents
            for i in range(prob_masks.shape[0]):
                last_valid_index = torch.where(~prob_masks[i])[0][-1]
                last_valid_indices.append(last_valid_index)

            for i, latents in enumerate(history_latents):
                last_valid_index = last_valid_indices[i]
                latents = latents[last_valid_index]
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents.unsqueeze(0), return_dict=False)[0].detach()
                images.append(self.image_processor.postprocess(image, output_type="pil"))

        return CustomDiffusionModelOutput(
            init_noise_latents=init_noise_latents,
            hidden_states_combineds=hidden_states_combineds,
            tembs=tembs,
            last_valid_indices=last_valid_indices,
            images=images,
            alphas=alphas,
            betas=betas,
            sigmas=sigmas,
            logprobs=logprobs,
            prob_masks=prob_masks,
        )

    def only_predict_logprobs(
        self,
        fix_sigmas: torch.Tensor,  # (bs, steps, ...)
        fix_hidden_states_combineds: torch.Tensor,  # (bs, steps, ...)
        fix_tembs: torch.Tensor, 
        fix_text_embeds: torch.Tensor = None,  # (bs, seq_len, embed_dim) for ViT predictor
    ):
        if fix_sigmas is None:
            raise ValueError("fix_sigmas must be provided")
        if fix_hidden_states_combineds is None:
            raise ValueError("fix_hidden_states_combineds must be provided")

        batch_size = fix_sigmas.shape[0]
        max_inference_steps = fix_sigmas.shape[1]

        sigma = torch.ones(batch_size, dtype=self.vae.dtype, device=self.vae.device)
        sigmas = [[] for _ in range(batch_size)]
        logprobs = [[] for _ in range(batch_size)]
        prob_masks = [[] for _ in range(batch_size)]

        fix_hidden_states_combineds = fix_hidden_states_combineds.to(self.vae.device).permute(1, 0, 2, 3, 4)
        fix_tembs = fix_tembs.permute(1, 0, 2)
        
        if fix_text_embeds is not None:
            fix_text_embeds = fix_text_embeds.to(self.vae.device)

        for step in range(max_inference_steps):
            fix_hidden_states_combined = fix_hidden_states_combineds[step]
            fix_temb = fix_tembs[step]
            
            # Call time predictor with appropriate inputs
            if self.use_vit_predictor and fix_text_embeds is not None:
                # For ViT with text embeddings, also use temb as timestep embedding
                time_pred = self.time_predictor(fix_hidden_states_combined, fix_text_embeds, fix_temb)
            else:
                # For CNN or when text embeddings not available
                time_pred = self.time_predictor(fix_hidden_states_combined, fix_temb)
            sigma_next = torch.zeros_like(sigma)
            for i, (alpha, beta) in enumerate(time_pred):
                beta_dist = torch.distributions.Beta(alpha, beta)
                # if now sigma is smaller than min_sigma, we should not get prob from beta_dist
                if sigma[i] < self.min_sigma:
                    sigma_next[i] = fix_sigmas[i][step]
                    sigmas[i].append(sigma_next[i])
                    logprobs[i].append(torch.tensor(0.0).to(self.vae.device))
                    prob_masks[i].append(torch.tensor(1))
                    continue
                else:
                    sigma_next[i] = fix_sigmas[i][step]
                    ratio = sigma_next[i] / sigma[i] if self.relative else sigma[i] - sigma_next[i]
                ratio = torch.clamp(ratio, min=self.epsilon, max=1 - self.epsilon)
                sigmas[i].append(sigma_next[i])
                prob = beta_dist.log_prob(ratio)
                logprobs[i].append(prob)
                if sigma[i] < self.min_sigma:
                    prob_masks[i].append(torch.tensor(1))
                else:
                    prob_masks[i].append(torch.tensor(0))

            sigma = sigma_next

        # the value will not influent the result
        INVALID_LOGPROB = 1.0
        logprobs = torch.stack([torch.stack(item) for item in logprobs])
        prob_masks = torch.stack([torch.stack(item) for item in prob_masks]).bool().to(logprobs.device)
        logprobs = torch.masked_fill(logprobs, prob_masks, INVALID_LOGPROB)

        return {"logprobs": logprobs}


class SD3PredictNextTimeStepModelRLOOWrapper(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        torch_dtype: torch.dtype = torch.float16,
        min_sigma: float = 0.01,
        pre_process: bool = False,
        init_alpha: float = 1.5,
        init_beta: float = 0.5,
        relative: bool = True,
        prediction_type: str = "alpha_beta",
        fsdp: list = [],
        max_inference_steps: int = 28,
        use_vit_predictor: bool = False,
        time_predictor_config: Optional[TimePredictorConfig] = None,
    ):
        super(SD3PredictNextTimeStepModelRLOOWrapper, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.agent_model = SD3PredictNextTimeStepModel(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            init_alpha=init_alpha,
            init_beta=init_beta,
            min_sigma=min_sigma,
            pre_process=pre_process,
            relative=relative,
            prediction_type=prediction_type,
            use_vit_predictor=use_vit_predictor,
            time_predictor_config=time_predictor_config,
        ).eval()

        self.relative = relative
        self.fsdp = fsdp
        self.max_inference_steps = max_inference_steps

        self.agent_model.requires_grad_(False)

        self.agent_model.time_predictor.train()
        self.agent_model.time_predictor.requires_grad_(True)

        # self.ref_alpha, self.ref_beta = 15.0, 1.5
        # self.ref_distribution = torch.distributions.Beta(self.ref_alpha, self.ref_beta)

    def rloo_repeat(self, data, rloo_k):
        """make the data repeat rloo_k times
        Args:
            data: dict of data
            rloo_k: int
        Returns:
            data: dict of data that is repeated rloo_k times
        """
        data["prompt"] = data["prompt"] * rloo_k
        for key in [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]:
            if key in data:
                size = [rloo_k] + [1] * (len(data[key].shape) - 1)
                data[key] = data[key].repeat(*size)
        return data

    def sample(self, inputs):
        """Generate model outputs step by step for inputs
        Args:
            inputs: dict of inputs
        Returns:
            outputs: dict of final outputs after sampling
        """
        if "3.5" in self.pretrained_model_name_or_path:
            inputs["guidance_scale"] = 3.5
        inputs["max_inference_steps"] = self.max_inference_steps
        if len(self.fsdp) > 0:
            with FullyShardedDataParallel.summon_full_params(self):
                outputs = self.agent_model(**inputs)
        else:
            outputs = self.agent_model(**inputs)
        # TODO: add reward model
        # rewards = self.reward(inputs, outputs)
        # outputs.update(rewards)
        return outputs

    def reward(self, inputs, outputs, reward_model, gamma=0.8, return_last_reward=False):
        """Given a batch of model inputs and outputs, provide the rewards of the outputs, using the final image in outputs
        Args:
            inputs: dict of inputs
            outputs: dict of outputs
            reward_model: reward model
            return_last_reward: whether to return the last reward
        Returns:
            rewards: tensor of rewards (bs, )
        """
        prompts = inputs.get("prompt", None)
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
                last_image = image[0]
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
        # outputs = self.agent_model(latents=outputs["init_noise_latents"], fix_sigmas=outputs["sigmas"], **inputs)
        if len(self.fsdp) > 0:
            with FullyShardedDataParallel.summon_full_params(self):
                outputs = self.agent_model.only_predict_logprobs(
                    fix_sigmas=outputs["sigmas"],
                    fix_hidden_states_combineds=outputs["hidden_states_combineds"],
                    fix_tembs=outputs["tembs"],
                )
        else:
            outputs = self.agent_model.only_predict_logprobs(
                fix_sigmas=outputs["sigmas"],
                fix_hidden_states_combineds=outputs["hidden_states_combineds"],
                fix_tembs=outputs["tembs"],
            )
        return outputs["logprobs"]

    def kl_divergence(self, outputs: CustomDiffusionModelOutput):
        """Given a batch of model outputs, provide the kl divergence of the outputs, using the alphas and betas in outputs
        Args:
            outputs: dict of outputs
        Returns:
            kl_divergence: tensor of kl divergence (bs, )
        """
        alphas = outputs["alphas"]
        betas = outputs["betas"]
        prob_masks = outputs["prob_masks"]
        kl_divergences = [[] for _ in range(len(alphas))]
        input_sigmas = F.pad(outputs["sigmas"][..., :-1], (1, 0), value=1.0)
        ref_alphas, ref_betas = get_ref_beta(input_sigmas)
        for i, (sub_alpha, sub_beta, ref_sub_alpha, ref_sub_beta) in enumerate(
            zip(alphas, betas, ref_alphas, ref_betas)
        ):
            for j, (alpha, beta, ref_alpha, ref_beta) in enumerate(
                zip(sub_alpha, sub_beta, ref_sub_alpha, ref_sub_beta)
            ):
                if prob_masks[i][j]:
                    kl_divergences[i].append(torch.tensor(0.0))
                else:
                    ref_distribution = torch.distributions.Beta(ref_alpha, ref_beta) if self.relative else torch.distributions.Beta(1.4, 11.2)
                    kl_divergences[i].append(
                        torch.distributions.kl_divergence(torch.distributions.Beta(alpha, beta), ref_distribution)
                    )
        return torch.tensor(kl_divergences)

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
    model = SD3PredictNextTimeStepModelRLOOWrapper(
        "models/stabilityai/stable-diffusion-3-medium-diffusers",
    ).cuda()
    inputs = {
        "prompt": [
            "a cat is holding a paper with 'hello world'",
            "a dog is holding a paper with 'hello world'",
        ],
        "max_inference_steps": 28,
        "guidance_scale": 7.0,
    }
    inputs = model.rloo_repeat(inputs, 2)
    outputs = model.sample(
        inputs=inputs,
    )
    kl_divergence = model.kl_divergence(
        outputs=outputs,
    )
    logprobs = model.logprobs(
        inputs=inputs,
        outputs=outputs,
    )
    rewards = model.reward(
        inputs=inputs,
        outputs=outputs,
    )
