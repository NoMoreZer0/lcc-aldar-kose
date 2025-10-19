# Training Framework Quick Start

Complete training framework for finetuning SDXL with **DreamBooth + LoRA** for Aldar Köse character consistency.

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create a directory with 10-30 images of Aldar Köse:

```bash
mkdir -p data/train/aldar_kose
# Add your images (JPG or PNG) to this directory
```

**Image requirements:**
- ✅ High quality (sharp, well-lit)
- ✅ Varied angles and expressions
- ✅ 1024x1024 or higher resolution
- ✅ Clear view of the character
- ❌ Avoid heavy filters or artifacts

### 3. Run Training

```bash
chmod +x examples/train_aldar_kose.sh
./examples/train_aldar_kose.sh
```

This will:
1. Generate class images for prior preservation
2. Train SDXL with DreamBooth + LoRA (500 steps)
3. Save checkpoints every 100 steps
4. Output to `output/aldar_kose_lora/`

**Training time:** ~30-60 minutes on a 24GB GPU

### 4. Use Your Trained Model

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load pipeline with your LoRA
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("output/aldar_kose_lora/final")

# Generate with trigger token "sks aldar"
image = pipe(
    "a photo of sks aldar in kazakh steppe at sunset, cinematic",
    num_inference_steps=30
).images[0]
```

---

## 📁 What Was Created

```
ml/
├── configs/
│   └── training.yaml              # Training configuration template
├── src/
│   ├── training/                  # Training framework
│   │   ├── trainer.py             # Main training loop
│   │   ├── dataset.py             # Data loaders
│   │   ├── utils.py               # Utilities (EMA, checkpointing)
│   │   └── lora.py                # LoRA implementation
│   └── cli/
│       ├── train.py               # Generic training CLI
│       ├── train_dreambooth.py    # DreamBooth-specific CLI
│       └── generate_class_images.py  # Prior preservation helper
├── examples/
│   ├── train_aldar_kose.sh       # Ready-to-use training script
│   └── generate_with_lora.py     # Inference example
├── TRAINING_README.md             # Comprehensive documentation
└── TRAINING_QUICKSTART.md         # This file
```

---

## 🎯 Training Methods

### Method 1: DreamBooth + LoRA (Recommended)

**Best for:** Character identity learning

```bash
python -m ml.src.cli.train_dreambooth \
  --instance_data_dir data/train/aldar_kose \
  --instance_prompt "a photo of sks aldar" \
  --output_dir output/aldar_lora \
  --use_lora \
  --lora_rank 64 \
  --max_train_steps 500
```

**Pros:**
- ✅ Low VRAM (~12-16GB)
- ✅ Fast training (30-60 min)
- ✅ Excellent character consistency
- ✅ Small output files (~50MB)

### Method 2: Generic LoRA

**Best for:** Style transfer, concept learning

```bash
python -m ml.src.cli.train \
  --config configs/training.yaml \
  --output_dir output/style_lora
```

Requires `data/train/metadata.jsonl`:
```jsonl
{"image_path": "image1.jpg", "prompt": "kazakh folk art style"}
{"image_path": "image2.jpg", "prompt": "traditional kazakh painting"}
```

---

## 🔧 Configuration

### Key Parameters

**Training Speed vs Quality:**
```yaml
# Fast (30 min, good quality)
max_train_steps: 400
learning_rate: 1.0e-4

# Balanced (60 min, great quality)
max_train_steps: 800
learning_rate: 5.0e-5

# Careful (2 hrs, maximum quality)
max_train_steps: 1500
learning_rate: 2.0e-5
train_text_encoder: true
```

**Memory Management:**
```yaml
# Low VRAM (8-12GB)
train_batch_size: 1
gradient_accumulation_steps: 8
use_8bit_adam: true
gradient_checkpointing: true

# High VRAM (24GB+)
train_batch_size: 4
gradient_accumulation_steps: 2
```

**LoRA Strength:**
```yaml
# Light adaptation
lora_rank: 16

# Character identity (recommended)
lora_rank: 64

# Complex multi-character scenes
lora_rank: 128
```

---

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

### Weights & Biases

```yaml
# In configs/training.yaml
logging:
  use_wandb: true
  wandb_project: "aldar-kose"
```

```bash
wandb login
python -m ml.src.cli.train --config configs/training.yaml
```

---

## ✅ Validation

### Check Training Quality

1. **View validation images** in `output/*/checkpoint-*/validation/`
2. **Check loss curves** in TensorBoard
3. **Test identity consistency:**

```python
python examples/generate_with_lora.py \
  --lora_path output/aldar_kose_lora/checkpoint-500 \
  --compare \
  --compare_prompt "a photo of sks aldar, portrait"
```

This generates side-by-side comparison (with vs without LoRA).

### Signs of Good Training

- ✅ Training loss decreases steadily
- ✅ Validation images show consistent character
- ✅ Character appears in varied poses/scenes
- ✅ Background changes appropriately with prompt

### Signs of Overfitting

- ❌ Validation loss increases
- ❌ Generated images copy training poses exactly
- ❌ Character only works in specific backgrounds
- ❌ Loss of flexibility in prompts

**Fix:** Reduce `max_train_steps` or `learning_rate`

---

## 🎨 Using Trained LoRA

### With Existing Pipeline

```python
# In your storyboard generation
from ml.src.diffusion.engine import SDXLEngine

engine = SDXLEngine(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda"
)

# Load LoRA
engine.pipe.load_lora_weights("output/aldar_kose_lora/final")

# Generate frames (will use LoRA automatically)
image = engine.txt2img(
    prompt="sks aldar in kazakh steppe, cinematic",
    height=1024,
    width=1024
)
```

### Generate Full Storyboard

```bash
python examples/generate_with_lora.py \
  --logline "Aldar Köse outsmarts a greedy merchant" \
  --frames 8 \
  --lora_path output/aldar_kose_lora/final \
  --instance_token "sks aldar" \
  --output_dir output/my_storyboard
```

### Combining Multiple LoRAs

```python
# Load character LoRA
pipe.load_lora_weights("output/aldar_kose_lora/final")

# Load style LoRA (weighted)
pipe.load_lora_weights(
    "output/kazakh_style_lora/final",
    adapter_name="style"
)

# Set weights
pipe.set_adapters(["character", "style"], [1.0, 0.7])
```

---

## 🐛 Troubleshooting

### Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--train_batch_size 1`
2. Enable CPU offload: `--enable_cpu_offload`
3. Use 8-bit Adam: `--use_8bit_adam`
4. Enable gradient checkpointing: `--gradient_checkpointing`
5. Reduce image resolution: `--resolution 768`

### Weak Character Identity

**Problem:** Generated character doesn't look like training images

**Solutions:**
1. Increase training steps: `--max_train_steps 800`
2. Train text encoder: `--train_text_encoder`
3. Increase LoRA rank: `--lora_rank 128`
4. Check trigger token in all prompts: `"sks aldar ..."`
5. Add more diverse training images

### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Enable xformers: `--enable_xformers`
2. Use mixed precision: `--mixed_precision fp16`
3. Cache latents: `cache_latents: true` in config
4. Use fewer validation steps: `--validation_steps 200`
5. Consider smaller LoRA rank: `--lora_rank 32`

---

## 📚 Next Steps

1. **Read full documentation:** [TRAINING_README.md](TRAINING_README.md)
2. **Experiment with parameters:** Adjust `max_train_steps`, `lora_rank`
3. **Try different prompts:** Test character in various scenes
4. **Combine with existing pipeline:** Integrate into storyboard generation
5. **Train multiple LoRAs:** Different characters, styles, or concepts

---

## 💡 Pro Tips

### Data Collection

1. **Quality > Quantity:** 15 great images > 50 mediocre ones
2. **Diversity matters:** Vary angles, expressions, lighting
3. **Clean backgrounds:** Remove distracting elements
4. **Consistent character:** Same outfit/appearance across images

### Training Strategy

1. **Start conservative:** 500 steps, check results, adjust
2. **Use validation:** Generate test images every 100 steps
3. **Save checkpoints:** Keep multiple versions to choose from
4. **Monitor loss:** Should decrease steadily (not plateau or spike)

### Prompt Engineering

1. **Always use trigger:** `"sks aldar"` must be in every prompt
2. **Describe scene separately:** `"sks aldar | in kazakh steppe"`
3. **Use negative prompts:** `"low quality, distorted, multiple people"`
4. **Test edge cases:** Unusual poses, lighting, compositions

---

## 🆘 Support

- **Documentation:** [TRAINING_README.md](TRAINING_README.md)
- **Examples:** See `examples/` directory
- **Logs:** Check `logs/` and `output/*/training_state.pt`
- **HuggingFace Docs:** https://huggingface.co/docs/diffusers/training/dreambooth

---

**Happy Training! 🚀**

Your trained LoRA will enable consistent Aldar Köse generation across all storyboard frames!
