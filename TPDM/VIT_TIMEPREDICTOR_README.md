# ViT TimePredictor for SD3/SD3.5

This directory contains an enhanced TimePredictor implementation using Vision Transformers (ViT) with text conditioning support for Stable Diffusion 3 and 3.5 models.

## Features

### TimePredictor Architectures
1. **CNN TimePredictor** (Original): Convolutional neural network-based predictor
2. **ViT TimePredictor** (New): Vision Transformer-based predictor with text conditioning
3. **Hybrid TimePredictor**: Wrapper that can switch between CNN and ViT architectures

### Key Improvements
- **Text Conditioning**: ViT TimePredictor can use text embeddings for better predictions
- **Timestep Conditioning**: Proper integration of timestep embeddings via AdaLN (Adaptive Layer Normalization)
- **Configurable Architecture**: Flexible configuration system for scaling up/down
- **Cross-Attention**: Optional cross-attention between visual and text features
- **Better Scalability**: ViT architecture scales better with model size
- **AdaLN Integration**: Timestep-modulated normalization for better temporal understanding

## Configuration

### ViT TimePredictor Config Options
```yaml
time_predictor_config:
  # Transformer architecture
  num_layers: 6          # Number of transformer layers
  num_heads: 8           # Number of attention heads
  hidden_size: 512       # Hidden dimension
  mlp_ratio: 4.0         # MLP expansion ratio
  dropout: 0.1           # Dropout rate
  attention_dropout: 0.1 # Attention dropout rate
  
  # Input/Output
  image_size: 64         # Input image size (after patch embedding)
  patch_size: 8          # Patch size for vision transformer
  in_channels: 3072      # Input channels (combined hidden states)
  text_embed_dim: 1536   # Text embedding dimension
  timestep_embed_dim: 1536  # Timestep embedding dimension
  projection_dim: 2      # Output dimension (alpha, beta)
  
  # Initialization
  init_alpha: 1.5        # Initial alpha value
  init_beta: 0.5         # Initial beta value
  epsilon: 1.0           # Numerical stability
  
  # Features
  use_text_conditioning: true     # Enable text conditioning
  use_timestep_conditioning: true # Enable timestep conditioning
  cross_attention: true           # Enable cross-attention
```

## Usage

### Training Scripts

1. **SD3.5 with ViT TimePredictor (Full Training)**:
   ```bash
   qsub scripts/launch_sd35_vit_train.sh
   ```

2. **SD3.5 with ViT TimePredictor (Test/Debug)**:
   ```bash
   qsub scripts/launch_sd35_vit_test.sh
   ```

3. **Flexible Training** (Choose model and predictor type):
   ```bash
   # SD3.5 with ViT
   qsub -v MODEL_TYPE=sd35,PREDICTOR_TYPE=vit scripts/launch_flexible_train.sh
   
   # SD3.5 with CNN
   qsub -v MODEL_TYPE=sd35,PREDICTOR_TYPE=cnn scripts/launch_flexible_train.sh
   
   # SD3 with ViT
   qsub -v MODEL_TYPE=sd3,PREDICTOR_TYPE=vit scripts/launch_flexible_train.sh
   ```

### Model Configurations

1. **SD3 + ViT TimePredictor**: `configs/models/sd3_pnt_vit.yaml`
2. **SD3.5 + ViT TimePredictor**: `configs/models/sd35_pnt_vit.yaml`
3. **SD3 + CNN TimePredictor**: `configs/models/sd3_pnt.yaml`

### Evaluation

```bash
# Evaluate with custom prompts
python scripts/eval_vit_timepredictor.py \
  --model_config configs/models/sd35_pnt_vit.yaml \
  --checkpoint path/to/checkpoint.pt \
  --prompts_file configs/prompts/test_prompts.json \
  --output_dir evaluation_results \
  --batch_size 4 \
  --max_steps 28

# Evaluate with single prompt
python scripts/eval_vit_timepredictor.py \
  --model_config configs/models/sd35_pnt_vit.yaml \
  --prompt "A beautiful sunset over mountains" \
  --output_dir evaluation_results
```

## Architecture Details

### ViT TimePredictor Components

1. **Patch Embedding**: Converts input image patches to tokens
2. **Positional Embedding**: Adds spatial position information
3. **Class Token**: Global representation token for final prediction
4. **Text Projection**: Projects text embeddings to hidden dimension
5. **Timestep Projection**: Projects timestep embeddings for AdaLN conditioning
6. **Transformer Blocks**: Self-attention, cross-attention, and AdaLN layers
7. **Prediction Head**: Final linear layer outputting alpha/beta parameters

### Text and Timestep Conditioning

The ViT TimePredictor uses both text and timestep embeddings:
1. **Text Cross-Attention**: Each transformer block can attend to text features
2. **Timestep AdaLN**: Adaptive Layer Normalization modulated by timestep embeddings
3. **Multi-Modal Integration**: Visual, text, and temporal information combined effectively

### Hybrid Architecture

The `HybridTimePredictor` allows switching between CNN and ViT:
```python
# ViT mode
predictor = HybridTimePredictor(config, use_vit=True)

# CNN mode
predictor = HybridTimePredictor(config, use_vit=False)
```

## File Structure

```
TPDM/
├── src/models/stable_diffusion_3/
│   └── modeling_sd3_pnt.py          # Main model implementation
├── configs/
│   ├── models/
│   │   ├── sd3_pnt_vit.yaml         # SD3 + ViT config
│   │   └── sd35_pnt_vit.yaml        # SD3.5 + ViT config
│   └── prompts/
│       ├── test_prompts.json        # Test prompts
│       └── artistic_prompts.json    # Artistic prompts
└── scripts/
    ├── launch_sd35_vit_train.sh     # SD3.5 ViT training
    ├── launch_sd35_vit_test.sh      # SD3.5 ViT testing
    ├── launch_flexible_train.sh     # Flexible training script
    └── eval_vit_timepredictor.py    # Evaluation script
```

## Performance Notes

### Memory Usage
- ViT TimePredictor uses more memory than CNN version
- Text conditioning adds additional memory overhead
- Consider reducing batch size or using gradient checkpointing

### Training Tips
- Start with smaller ViT configurations (fewer layers/heads)
- Use cosine learning rate schedule for ViT
- Consider warming up the learning rate
- Text conditioning may require lower learning rates

### Scaling Guidelines
- **Small**: 4-6 layers, 6-8 heads, 384-512 hidden size
- **Medium**: 6-8 layers, 8-12 heads, 512-768 hidden size  
- **Large**: 8-12 layers, 12-16 heads, 768-1024 hidden size

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or model size
2. **Slow Training**: Check if cross-attention is needed
3. **Poor Convergence**: Try different learning rates or schedulers
4. **Text Conditioning Not Working**: Ensure text embeddings are properly passed

### Debug Mode
Use the test training script for quick debugging:
```bash
qsub scripts/launch_sd35_vit_test.sh
```

This runs with smaller batch sizes and shorter training for fast iteration.
