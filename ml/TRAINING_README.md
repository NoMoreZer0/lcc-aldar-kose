# Training Framework for SDXL Diffusion Model

Comprehensive training framework for finetuning the Stable Diffusion XL model with support for:
- **DreamBooth** for character identity learning
- **LoRA** for parameter-efficient finetuning
- **Full finetuning** for maximum control
- **Mixed precision training** (FP16/BF16)
- **Gradient accumulation** for large effective batch sizes
- **EMA (Exponential Moving Average)** for stable training
- **Distributed training** support

---

## Quick Start

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
pip install peft accelerate bitsandbytes safetensors
```

### 2. Prepare Your Data

For **DreamBooth character training** (recommended for Aldar Köse):

```
data/
├── train/
│   ├── aldar_kose/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── metadata.jsonl
└── val/
    ├── aldar_kose/
    └── metadata.jsonl
```

**metadata.jsonl** format:
```jsonl
{"image_path": "aldar_kose/image_001.jpg", "prompt": "a photo of sks aldar wearing traditional chapan and kalpak"}
{"image_path": "aldar_kose/image_002.jpg", "prompt": "sks aldar smiling in kazakh steppe setting"}
```

**Key points:**
- Use a **unique token** (e.g., `sks`) + character name to bind identity
- Collect **10-30 diverse images** (varied angles, lighting, poses)
- Images should be high quality, 1024x1024 recommended
- Include varied backgrounds and expressions

### 3. Configure Training

Edit [`configs/training.yaml`](configs/training.yaml) or create your own:

```yaml
# Minimal DreamBooth LoRA config
model:
  base_model_id: "stabilityai/stable-diffusion-xl-base-1.0"
  finetune_method: "lora"  # or "full" or "dreambooth"
  resolution: 1024
  gradient_checkpointing: true
  enable_xformers: true

lora:
  rank: 64
  alpha: 64
  target_modules:
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out.0"

dataset:
  train_data_dir: "data/train"
  metadata_file: "metadata.jsonl"
  image_column: "image_path"
  caption_column: "prompt"

training:
  output_dir: "output/aldar_kose_lora"
  train_batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 10
  max_train_steps: 500  # DreamBooth: start with 500 steps
  learning_rate: 1.0e-4
  mixed_precision: "fp16"
  lr_scheduler: "constant"
  lr_warmup_steps: 0
```

### 4. Train the Model

#### Basic Training (LoRA)
```bash
python -m ml.src.cli.train \
  --config ml/configs/training.yaml \
  --output_dir output/aldar_kose_lora
```

#### DreamBooth-specific Training
```bash
python -m ml.src.cli.train_dreambooth \
  --instance_data_dir data/train/aldar_kose \
  --instance_prompt "a photo of sks aldar" \
  --class_prompt "a photo of a person" \
  --output_dir output/aldar_dreambooth \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 500 \
  --use_lora
```

#### Resume from Checkpoint
```bash
python -m ml.src.cli.train \
  --config ml/configs/training.yaml \
  --resume_from_checkpoint output/aldar_kose_lora/checkpoint-500
```

---

## Training Methods Comparison

| Method | VRAM | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| **LoRA** | ~12GB | Fast | Good | Character identity, style transfer |
| **DreamBooth** | ~16GB | Medium | Excellent | Character identity (gold standard) |
| **Full Finetune** | ~40GB | Slow | Best | Complete model adaptation |

---

## DreamBooth Training (Recommended for Character Learning)

DreamBooth is the best method for teaching SDXL a specific character like Aldar Köse.

### Data Preparation

1. **Collect 10-30 images** of Aldar Köse:
   - Varied angles (front, side, 3/4 view)
   - Different expressions (smiling, serious, thoughtful)
   - Various poses and gestures
   - Multiple backgrounds (steppe, yurt, marketplace)
   - Different lighting conditions

2. **Choose a unique token**: `sks` or `ohwx` (rare tokens)

3. **Create captions**:
   ```
   "a photo of sks aldar wearing traditional chapan"
   "sks aldar smiling in kazakh steppe"
   "portrait of sks aldar with kalpak hat"
   ```

### Training Parameters

**Conservative (prevents overfitting):**
```yaml
max_train_steps: 400-500
learning_rate: 1.0e-4
train_batch_size: 1
gradient_accumulation_steps: 4
```

**Aggressive (stronger identity):**
```yaml
max_train_steps: 800-1000
learning_rate: 5.0e-5
train_batch_size: 2
train_text_encoder: true  # Trains CLIP text encoders
```

### Prior Preservation (Optional)

Prevents "model forgetting" what a generic person looks like:

```yaml
prior_preservation:
  enabled: true
  weight: 1.0
  class_data_dir: "data/class_images"
  num_class_images: 200
  class_prompt: "a photo of a person"
```

Generate class images first:
```bash
python -m ml.src.cli.generate_class_images \
  --class_prompt "a photo of a person" \
  --num_images 200 \
  --output_dir data/class_images
```

---

## LoRA Training (Fast & Memory Efficient)

LoRA adds small adapter layers instead of finetuning the entire model.

### Advantages
- ✅ **Low VRAM**: ~12GB for SDXL
- ✅ **Fast training**: 2-4x faster than full finetuning
- ✅ **Small files**: LoRA weights are ~50MB vs 6GB for full model
- ✅ **Composable**: Can combine multiple LoRAs

### LoRA Configuration

```yaml
lora:
  rank: 64           # Higher = more capacity (16-128)
  alpha: 64          # Usually same as rank
  target_modules:    # Which layers to adapt
    - "to_q"         # Query projection
    - "to_k"         # Key projection
    - "to_v"         # Value projection
    - "to_out.0"     # Output projection
    - "add_k_proj"   # SDXL additional projections
    - "add_v_proj"
  dropout: 0.0       # LoRA dropout (0.0-0.1)
```

**Rank guidelines:**
- `rank=16-32`: Light style transfer
- `rank=64`: Character identity (recommended)
- `rank=128`: Complex concepts, multiple characters

### Training Command

```bash
python -m ml.src.cli.train \
  --config ml/configs/training.yaml
```

### Inference with LoRA

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load your LoRA
pipe.load_lora_weights("output/aldar_kose_lora/checkpoint-500")

# Generate with trigger token
image = pipe(
    "a photo of sks aldar in kazakh steppe at sunset",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("aldar_generated.png")
```

---

## Memory Optimization

For GPUs with **limited VRAM** (8-16GB):

```yaml
model:
  gradient_checkpointing: true
  enable_xformers: true

training:
  mixed_precision: "fp16"
  train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 8
  use_8bit_adam: true
  enable_cpu_offload: true
```

**Additional optimizations:**
```bash
# Use 8-bit optimizer
pip install bitsandbytes

# Enable xformers
pip install xformers

# Cache latents to disk (saves VAE encoding time)
cache_latents: true
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs
```

Metrics to watch:
- **Training loss**: Should decrease steadily
- **Validation loss**: Should track training loss
- **Learning rate**: Check warmup and decay schedules

### Weights & Biases

```yaml
logging:
  use_wandb: true
  wandb_project: "aldar-kose-training"
  wandb_run_name: "dreambooth-lora-v1"
```

```bash
# Login to W&B
wandb login

# Training will auto-log to W&B
python -m ml.src.cli.train --config configs/training.yaml
```

---

## Validation & Quality Checks

### Automatic Validation

The trainer runs validation every N steps:

```yaml
validation:
  validation_steps: 100
  num_validation_samples: 4
  validation_prompts:
    - "a photo of sks aldar in traditional kazakh clothing"
    - "portrait of sks aldar smiling"
    - "sks aldar in kazakh steppe landscape"
    - "cinematic shot of sks aldar at sunset"
```

### Manual Testing

```python
from ml.src.diffusion.engine import SDXLEngine

engine = SDXLEngine(
    model_id="output/aldar_kose_lora/checkpoint-500",
    device="cuda"
)

image = engine.txt2img(
    prompt="a photo of sks aldar wearing traditional chapan",
    height=1024,
    width=1024,
    num_steps=30
)
```

### Identity Consistency Metrics

The framework includes automatic identity evaluation:

```yaml
consistency:
  enable_identity_loss: true
  identity_loss_weight: 0.1
  face_model: "buffalo_l"
```

Metrics computed:
- **Face similarity**: Cosine similarity of face embeddings
- **SSIM**: Background consistency
- **LPIPS**: Perceptual similarity

---

## Best Practices

### Character Identity Training (DreamBooth)

1. **Data quality > quantity**: 15 great images > 50 mediocre ones
2. **Trigger token**: Always use rare token (`sks`, `ohwx`) + character name
3. **Diverse shots**: Vary angles, expressions, lighting, backgrounds
4. **Start conservative**: 400-500 steps, check results, increase if needed
5. **Monitor overfitting**: If validation loss increases, stop training
6. **Text encoder**: Add `train_text_encoder: true` if identity is weak

### LoRA Training

1. **Rank selection**: Start with 64, decrease if overfitting
2. **Learning rate**: 1e-4 is safe, can go to 5e-5 for finer control
3. **Target modules**: More modules = stronger effect but slower training
4. **Alpha = Rank**: Good default, can adjust for scaling

### General Tips

1. **Save checkpoints frequently**: Every 100-500 steps
2. **Use EMA**: Smooths training, often better results
3. **Mixed precision**: Always use FP16 for SDXL (saves memory + faster)
4. **Gradient accumulation**: Simulate larger batches without VRAM cost
5. **Validation prompts**: Include edge cases and difficult compositions

---

## Troubleshooting

### Out of Memory (OOM)

```yaml
# Enable all memory optimizations
model:
  gradient_checkpointing: true
  enable_xformers: true

training:
  train_batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: "fp16"
  use_8bit_adam: true
  enable_cpu_offload: true

dataset:
  cache_latents: true
```

### Overfitting

**Symptoms:**
- Training loss decreases, validation loss increases
- Generated images look exactly like training data
- Character only works in training poses/backgrounds

**Solutions:**
- Reduce `max_train_steps` (try 300-400)
- Lower `learning_rate` (try 5e-5)
- Add more diverse training images
- Enable dropout: `lora.dropout: 0.1`
- Use prior preservation for DreamBooth

### Underfitting / Weak Identity

**Symptoms:**
- Character doesn't look like training images
- Trigger token has no effect
- Generic results

**Solutions:**
- Increase `max_train_steps` (try 800-1000)
- Increase LoRA rank: `lora.rank: 128`
- Enable text encoder training: `text_encoder.train_text_encoder: true`
- Check trigger token is in all training captions
- Increase `learning_rate` slightly (try 2e-4)

### Slow Training

```yaml
# Speed optimizations
model:
  enable_xformers: true

dataset:
  cache_latents: true  # Pre-encode images
  num_workers: 4       # Parallel data loading

training:
  mixed_precision: "fp16"  # or "bf16" on A100
```

---

## File Structure

```
ml/
├── configs/
│   ├── training.yaml              # Main training config
│   └── dreambooth.yaml            # DreamBooth-specific config
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── dataset.py             # Dataset loaders
│   │   ├── utils.py               # EMA, checkpointing, metrics
│   │   └── lora.py                # LoRA implementation
│   └── cli/
│       ├── train.py               # Training CLI
│       └── train_dreambooth.py    # DreamBooth CLI
├── data/
│   ├── train/                     # Training images
│   ├── val/                       # Validation images
│   └── class_images/              # Prior preservation images
└── output/
    └── checkpoints/               # Saved models
```

---

## Example Workflows

### Workflow 1: Train Aldar Köse with DreamBooth LoRA

```bash
# 1. Prepare data (15-20 images of Aldar Köse)
mkdir -p data/train/aldar_kose
# Add images to data/train/aldar_kose/

# 2. Create metadata
python -m ml.src.cli.create_metadata \
  --image_dir data/train/aldar_kose \
  --instance_prompt "a photo of sks aldar" \
  --output data/train/metadata.jsonl

# 3. Train
python -m ml.src.cli.train \
  --config ml/configs/training.yaml \
  --output_dir output/aldar_lora

# 4. Test
python -m ml.src.cli.generate_storyboard \
  --logline "Aldar Köse tricks a greedy merchant" \
  --frames 6 \
  --lora_path output/aldar_lora/checkpoint-500
```

### Workflow 2: Incremental Training

```bash
# Train for 500 steps
python -m ml.src.cli.train --config configs/training.yaml

# Evaluate results, if needed continue:
python -m ml.src.cli.train \
  --config configs/training.yaml \
  --resume_from_checkpoint output/aldar_lora/checkpoint-500
```

---

## Advanced: Custom Training Scripts

You can use the training framework programmatically:

```python
from ml.src.training import DiffusionTrainer, StoryboardDataset, collate_fn
from torch.utils.data import DataLoader
import yaml

# Load config
with open("configs/training.yaml") as f:
    config = yaml.safe_load(f)

# Initialize trainer
trainer = DiffusionTrainer(config)

# Create dataset
train_dataset = StoryboardDataset(
    data_dir="data/train",
    tokenizer=trainer.tokenizer,
    resolution=1024
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

# Train
trainer.train(train_loader)
```

---

## References

- [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `logs/` directory
3. Inspect checkpoints in `output/` directory
4. Validate data format matches examples

Happy training! 🚀
