# Flux LayerDiffuse Usage Guide

## 🚀 Quick Start

### Step 1: Prepare Your Models

1. **Download Flux Model**: Place Flux.1-dev model in `ComfyUI/models/diffusion_models/`
2. **Download LayerDiffuse LoRA**: Place `layerlora.safetensors` in `ComfyUI/models/loras/`
3. **Download TransparentVAE**: Place `TransparentVAE.pth` in `ComfyUI/models/vae/`

### Step 2: Basic Workflow

```
1. [Load Diffusion Model] → Load your Flux model
2. [Load LoRA] → Load layerlora.safetensors (strength: 1.0)
3. [CLIP Text Encode] → Your prompt (e.g., "glass bottle, transparent")
4. [Empty Latent (Transparent)] → Set your desired size (1024x1024)
5. [Flux LayerDiffuse Standalone Loader] → Load TransparentVAE.pth
6. [Flux LayerDiffuse Sampler] → Connect all inputs
7. [Save Image] → Save your transparent image
```

**Note**: The new Standalone Loader doesn't need a separate VAE input - it loads the TransparentVAE directly!

## 📋 Node Reference

### 🔧 Flux LayerDiffuse Standalone Loader
**Purpose**: Loads TransparentVAE directly from checkpoint file

**Inputs**:
- `transparent_vae_checkpoint`: Filename (e.g., "TransparentVAE.pth")
- `alpha`: Transparency strength (default: 300.0)
- `latent_channels`: Usually 16 for Flux
- `dtype`: Precision (bfloat16 recommended)

**Output**: `transparent_vae` for use in other nodes

### ℹ️ Flux LayerDiffuse Info
**Purpose**: Check file locations and provide setup guidance

**Inputs**:
- `action`: "check_files" or "setup_guide"

**Output**: Information text

### 🎨 Flux LayerDiffuse Sampler
**Purpose**: Samples using ComfyUI's system + TransparentVAE decoding

**Inputs**:
- `model`: Flux model with LoRA applied
- `transparent_vae`: From VAE Loader
- `positive`: Text conditioning
- `negative`: Negative conditioning
- `latent_image`: From Empty Latent (Transparent)
- Standard sampling parameters (steps, cfg, etc.)

**Outputs**: 
- `latent`: Standard latent for further processing
- `transparent_image`: Ready-to-save transparent image

### 🔍 Flux LayerDiffuse Decoder
**Purpose**: Decode existing latents to transparent images

**Inputs**:
- `transparent_vae`: From VAE Loader
- `samples`: Latent from any sampler
- `use_augmentation`: Quality vs speed trade-off

**Output**: `transparent_image`

### 📋 Empty Latent (Transparent)
**Purpose**: Creates properly sized latent for Flux transparent generation

**Inputs**:
- `width`, `height`: Image dimensions
- `batch_size`: Number of images

**Output**: `latent` for sampling

## 🎯 Workflow Examples

### Example 1: Simple Generation
```
Load Diffusion Model (Flux) → Load LoRA (layerlora.safetensors) → 
CLIP Text Encode ("glass bottle") → 
Empty Latent (Transparent) (1024x1024) →
Load VAE → Flux LayerDiffuse VAE Loader →
Flux LayerDiffuse Sampler → Save Image
```

### Example 2: Using Standard KSampler
```
Load Diffusion Model (Flux) → Load LoRA → KSampler →
Load VAE → Flux LayerDiffuse VAE Loader → 
Flux LayerDiffuse Decoder → Save Image
```

## 💡 Tips & Best Practices

### Prompts for Transparent Images
- ✅ Good: "glass bottle", "crystal sculpture", "transparent object"
- ✅ Good: "wine glass, elegant, transparent"
- ❌ Avoid: Background descriptions, "on white background"

### Model Settings
- **LoRA Strength**: 1.0 (full strength recommended)
- **CFG Scale**: 3.5-7.0 (lower values often work better)
- **Steps**: 20-50 (more steps = better quality)
- **Sampler**: euler, dpmpp_2m work well

### Performance
- **Fast**: `use_augmentation: false`, fewer steps
- **Quality**: `use_augmentation: true`, more steps
- **Memory**: Use bfloat16, smaller batch sizes

## 🔧 Troubleshooting

### "TransparentVAE weights not found"
- Check file is in `ComfyUI/models/vae/TransparentVAE.pth`
- Or use full path in the input field

### "Model not compatible"
- Ensure you're using a Flux.1-dev model
- Check that LoRA is properly loaded

### Poor transparency quality
- Increase CFG scale
- Enable augmentation
- Use more sampling steps
- Check prompt doesn't include backgrounds

### Memory issues
- Reduce image size
- Use float16 instead of bfloat16
- Disable augmentation
- Reduce batch size

## 📁 File Locations

Place your files in these ComfyUI directories:

```
ComfyUI/
├── models/
│   ├── diffusion_models/    # Flux models here (.safetensors)
│   ├── loras/               # layerlora.safetensors here
│   └── vae/                 # TransparentVAE.pth here
└── custom_nodes/
    └── denrakeiw_nodes/     # This node pack
```

## 🎨 Advanced Usage

### Batch Processing
Use `batch_size > 1` in Empty Latent (Transparent) for multiple images

### Custom Sizes
Flux works best with multiples of 64. Common sizes:
- 1024x1024 (square)
- 1024x768 (4:3)
- 1152x896 (landscape)
- 896x1152 (portrait)

### Quality Control
- Higher `alpha` values = stronger transparency effect
- `use_augmentation: true` = better quality, slower
- More sampling steps = better detail

## 🆘 Support

If you encounter issues:
1. Check ComfyUI console for error messages
2. Verify all files are in correct locations
3. Restart ComfyUI after installing
4. Test with simple prompts first
