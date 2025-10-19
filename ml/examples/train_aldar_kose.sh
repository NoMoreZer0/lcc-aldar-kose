#!/bin/bash
# Example: Train SDXL DreamBooth LoRA for Aldar Köse character
#
# This script demonstrates how to train a character-specific model
# for consistent generation of Aldar Köse across storyboard frames.
#
# Prerequisites:
# 1. Prepare 10-30 images of Aldar Köse in data/train/aldar_kose/
# 2. Install dependencies: pip install -r requirements.txt
# 3. Have at least 16GB GPU VRAM (or use memory optimizations below)

set -e  # Exit on error

# Configuration
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID="madebyollin/sdxl-vae-fp16-fix"
INSTANCE_DIR="data/train/aldar_kose"
OUTPUT_DIR="output/aldar_kose_lora"
INSTANCE_PROMPT="a photo of sks aldar"
CLASS_PROMPT="a photo of a person"
CLASS_DIR="data/class_images"

# Training parameters
RESOLUTION=1024
TRAIN_STEPS=500
LEARNING_RATE=1e-4
BATCH_SIZE=1
GRAD_ACCUM=4
LORA_RANK=64

# Validation
VALIDATION_PROMPT="a portrait of sks aldar wearing traditional kazakh chapan and kalpak, cinematic lighting"
VALIDATION_STEPS=100

echo "=== Training Aldar Köse DreamBooth LoRA ==="
echo "Instance images: $INSTANCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Training steps: $TRAIN_STEPS"
echo ""

# Check if instance images exist
if [ ! -d "$INSTANCE_DIR" ]; then
    echo "Error: Instance directory not found: $INSTANCE_DIR"
    echo "Please create it and add 10-30 images of Aldar Köse"
    exit 1
fi

NUM_IMAGES=$(find "$INSTANCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
if [ "$NUM_IMAGES" -lt 5 ]; then
    echo "Warning: Found only $NUM_IMAGES images. Recommended: 10-30 images"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
else
    echo "Found $NUM_IMAGES instance images ✓"
fi

# Generate class images for prior preservation (optional but recommended)
if [ ! -d "$CLASS_DIR" ] || [ $(ls -1 "$CLASS_DIR"/*.png 2>/dev/null | wc -l) -lt 100 ]; then
    echo ""
    echo "Generating class images for prior preservation..."
    python -m ml.src.cli.generate_class_images \
        --class_prompt "$CLASS_PROMPT" \
        --output_dir "$CLASS_DIR" \
        --num_images 200 \
        --resolution $RESOLUTION \
        --batch_size 2
    echo "Class images generated ✓"
fi

echo ""
echo "Starting DreamBooth training..."
echo ""

# Main training command
python -m ml.src.cli.train_dreambooth \
    --pretrained_model_name_or_path "$MODEL_ID" \
    --pretrained_vae_model_name_or_path "$VAE_ID" \
    --instance_data_dir "$INSTANCE_DIR" \
    --instance_prompt "$INSTANCE_PROMPT" \
    --class_data_dir "$CLASS_DIR" \
    --class_prompt "$CLASS_PROMPT" \
    --with_prior_preservation \
    --prior_loss_weight 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --max_train_steps $TRAIN_STEPS \
    --use_lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --mixed_precision "fp16" \
    --gradient_checkpointing \
    --enable_xformers \
    --use_8bit_adam \
    --checkpointing_steps $VALIDATION_STEPS \
    --validation_prompt "$VALIDATION_PROMPT" \
    --validation_steps $VALIDATION_STEPS \
    --seed 42

echo ""
echo "=== Training Complete! ==="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To use your trained LoRA:"
echo ""
echo "  from diffusers import StableDiffusionXLPipeline"
echo "  import torch"
echo ""
echo "  pipe = StableDiffusionXLPipeline.from_pretrained("
echo "      \"$MODEL_ID\","
echo "      torch_dtype=torch.float16"
echo "  ).to(\"cuda\")"
echo ""
echo "  pipe.load_lora_weights(\"$OUTPUT_DIR/final\")"
echo ""
echo "  image = pipe("
echo "      \"$INSTANCE_PROMPT in kazakh steppe at sunset\","
echo "      num_inference_steps=30"
echo "  ).images[0]"
echo ""
echo "Remember to use the trigger: '$INSTANCE_PROMPT' in your prompts!"
