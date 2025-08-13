# Denrakeiw Nodes

A comprehensive custom node pack for ComfyUI that provides utility nodes for image generation, manipulation, **Flux LayerDiffuse transparent image generation**, and **advanced latent space color tools**.

## Nodes

### Color Generator Node

![Screenshot 2025-07-25 175434](https://github.com/user-attachments/assets/ef53eb14-8e10-40df-9909-65825a425017)


Generates solid color images based on predefined colors.

**Features:**
- Width and height input fields (1-8192 pixels)
- Dropdown menu with 130+ predefined colors
- Outputs:
  - Solid color image
  - Color name as string
  - Hex code as string

**Available Color Categories:**
- **Basic Colors**: Red, Green, Blue, Yellow, Cyan, Magenta, White, Black
- **Grays**: Gray, Light Gray, Dark Gray, Dim Gray, Silver, Gainsboro, WhiteSmoke
- **Reds**: Dark Red, Crimson, Fire Brick, Indian Red, Light Coral, Salmon, Coral, Tomato, Orange Red, Maroon
- **Oranges**: Orange, Dark Orange, Peach, Papaya Whip, Moccasin, Peach Puff, Khaki
- **Yellows**: Light Yellow, Lemon Chiffon, Cornsilk, Gold, Goldenrod
- **Greens**: Lime, Lime Green, Lawn Green, Chartreuse, Spring Green, Forest Green, Sea Green, Olive, Yellow Green
- **Cyans/Teals**: Aqua, Dark Cyan, Teal, Turquoise, Aqua Marine, Cadet Blue
- **Blues**: Light Blue, Powder Blue, Steel Blue, Cornflower Blue, Deep Sky Blue, Royal Blue, Navy, Midnight Blue
- **Purples/Violets**: Purple, Indigo, Slate Blue, Blue Violet, Dark Orchid, Orchid, Violet, Plum, Lavender
- **Pinks**: Pink, Light Pink, Hot Pink, Deep Pink
- **Browns**: Brown, Saddle Brown, Sienna, Chocolate, Peru, Sandy Brown, Tan, Wheat
- **Special Colors**: Beige, Floral White, Ghost White, Honeydew, Ivory, Azure, Snow, Mint Cream

### 📁 Load Image Sequence
Loads images from a folder one by one, incrementing through the sequence with each workflow execution.

**Features:**
- Sequential image loading (one image per execution)
- Automatic incrementing through folder contents
- Robust error handling (skips corrupted images)
- Multiple sorting methods (alphabetical, date, size)
- Loop option (restart from beginning when reaching end)
- State management (remembers position between executions)
- **Outputs:**
  - Current image
  - Filename (e.g., "01.png")
  - TXT filename (e.g., "01.txt")
  - Current index (1-based)
  - Total image count

**Advantages over Load Image Batch:**
- ✅ Low memory usage (loads only one image at a time)
- ✅ No memory overflow issues
- ✅ Automatic progression through image sequence
- ✅ Better error handling for corrupted files

### 📊 Load Image Sequence Info
Displays information about the current image sequence state.

**Features:**
- Shows folder contents and current position
- Lists found image files
- Displays next image to be loaded
- Useful for debugging and monitoring progress

---

## 🎨 Latent Color Tools

### 🎨 Latent Color Match
Advanced color matching between latents using multiple algorithms, working directly in latent space for maximum efficiency.

**Features:**
- **Multiple color matching methods:**
  - **Cubiq-based methods** with kornia (LAB, YCbCr, LUV, YUV, XYZ, RGB)
  - **Advanced algorithms** with color-matcher (hm-mkl-hm, mkl, hm, reinhard, mvgd, hm-mvgd-hm)
- **Real-time processing** directly in latent space
- **Batch processing** support for efficiency
- **Factor control** (0.0-3.0) for effect strength
- **Automatic tensor shape handling** (4D and 5D tensors)
- **GPU acceleration** with full CUDA support

**Advantages over image-based color matching:**
- ⚡ **10x faster** - No VAE encoding/decoding needed
- 💾 **50% less VRAM** usage
- ✅ **No quality loss** from VAE artifacts
- 🔄 **Seamless integration** with latent workflows

### 🎨 Latent Color Match (Simple)
Simplified version with basic color matching methods for quick adjustments.

**Features:**
- Mean/std matching across all channels
- Channel-wise mean/std matching
- Lightweight and fast processing
- **Outputs:** Color-matched latent

### 🎛️ Latent Image Adjust
Complete image adjustment suite working directly in latent space.

**Features:**
- **Brightness** (-1.0 to 1.0) - Additive brightness adjustment
- **Contrast** (0.0 to 3.0) - Multiplicative contrast around mean
- **Hue** (-180° to 180°) - Color tone shifting with HSV conversion
- **Saturation** (0.0 to 3.0) - Color intensity adjustment
- **Sharpness** (0.0 to 3.0) - Unsharp masking and blur effects
- **Batch processing** for multiple latents
- **Device selection** (auto/CPU/GPU)

**Technical Benefits:**
- **Direct latent manipulation** - No image conversion needed
- **Kornia integration** for professional color space operations
- **Memory efficient** processing
- **Anti-aliasing** to prevent artifacts

---

## 🔥 Flux LayerDiffuse Nodes

### 🔧 Flux LayerDiffuse Standalone Loader
Loads the TransparentVAE model for transparent image generation.

**Features:**
- Loads TransparentVAE.pth model
- Requires standard VAE input for compatibility
- Configurable alpha, latent channels, and dtype
- **Outputs:** TRANSPARENT_VAE for decoder

### 🔍 Flux LayerDiffuse Decoder (Simple)
Decodes latents to transparent RGBA images.

**Features:**
- Takes latents from any sampler (KSampler, etc.)
- Uses TransparentVAE for transparent decoding
- Optional augmentation for better quality
- **Outputs:** Transparent IMAGE (RGBA format)

### 💾 Save Transparent Image
Saves transparent images as PNG files with proper alpha channel support.

**Features:**
- Saves real PNG files with transparency
- Automatic timestamped filenames
- Embeds metadata directly in PNG
- Shows saved image in ComfyUI interface
- **Outputs:** Displays image in node

### 👁️ Preview Transparent Image
Previews transparent images with different background options.

**Features:**
- Multiple background options: checkerboard, white, black, transparent
- Shows transparency correctly
- ComfyUI-compatible preview
- **Outputs:** Displays composited image in node

### 📊 Transparent Image Info
Analyzes transparent images and provides detailed information.

**Features:**
- Analyzes alpha channel statistics
- Shows image dimensions and format
- Transparency quality assessment
- **Outputs:** Detailed info string

### 🔍 Conditioning Inspector
Inspects CLIP conditioning tensors to verify Flux compatibility.

**Features:**
- Shows tensor dimensions and format
- Checks Flux compatibility
- Provides recommendations
- **Outputs:** Conditioning passthrough + info string

### 🔧 Flux LayerDiffuse Conditioning Fix
Fixes conditioning tensor dimensions for Flux compatibility (usually not needed with DualCLIPLoader type="flux").

**Features:**
- Converts SD conditioning to Flux format
- Adjustable target sequence length
- **Outputs:** Fixed conditioning

### ⭕ Flux LayerDiffuse Empty Conditioning
Creates empty conditioning for negative prompts.

**Features:**
- Generates proper empty conditioning
- Flux-compatible dimensions
- **Outputs:** Empty conditioning

### 🩺 Flux LayerDiffuse Troubleshooter
Provides solutions for common Flux LayerDiffuse issues.

**Features:**
- Diagnoses tensor dimension errors
- Provides step-by-step solutions
- Troubleshooting guides
- **Outputs:** Solution text

### 📖 Flux LayerDiffuse Workflow Helper
Provides workflow instructions and tips.

**Features:**
- Basic workflow guide
- Advanced workflow tips
- Model requirements
- **Outputs:** Instruction text

### ℹ️ Flux LayerDiffuse Info
Shows information about required models and file locations.

**Features:**
- Lists required model files
- Shows file locations and sizes
- Download links and instructions
- **Outputs:** Info text

---

## 📁 Model Installation

### Required Models for Flux LayerDiffuse:

1. **Flux Model** (place in `ComfyUI/models/diffusion_models/`):
   - `flux1-dev-fp8.safetensors` or `flux1-dev.safetensors`
   - Download from: [Hugging Face - Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

2. **LayerDiffuse LoRA** (place in `ComfyUI/models/loras/`):
   - `layerlora.safetensors` (included in this package)
   - Or download from: [Hugging Face - Flux LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

3. **TransparentVAE** (place in `ComfyUI/models/vae/`):
   - `TransparentVAE.pth` (included in this package)
   - Or download from: [Hugging Face - Flux LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

4. **CLIP Models** (place in `ComfyUI/models/clip/`):
   - `clip_l.safetensors`
   - `t5xxl_fp8_e4m3fn.safetensors`
   - Download from: [Hugging Face - CLIP](https://huggingface.co/comfyanonymous/flux_text_encoders)

5. **Standard VAE** (place in `ComfyUI/models/vae/`):
   - `ae.safetensors` (Flux VAE)
   - Download from: [Hugging Face - Flux VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev)

### File Structure:
```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── flux1-dev-fp8.safetensors
│   ├── loras/
│   │   └── layerlora.safetensors
│   ├── vae/
│   │   ├── TransparentVAE.pth
│   │   └── ae.safetensors
│   └── clip/
│       ├── clip_l.safetensors
│       └── t5xxl_fp8_e4m3fn.safetensors
└── custom_nodes/
    └── denrakeiw_nodes/
```

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:
   ```
   ComfyUI/custom_nodes/denrakeiw_nodes/
   ```

2. Restart ComfyUI

3. The nodes will appear under the "denrakeiw/image" category

## Requirements

### For Color Generator:
- ComfyUI
- torch
- numpy
- PIL (Pillow)

### For Latent Color Tools:
- ComfyUI
- torch
- numpy
- kornia>=0.6.0 (for advanced color space conversions)
- color-matcher>=0.2.0 (for professional color matching algorithms)

### For Flux LayerDiffuse:
- ComfyUI
- torch
- numpy
- PIL (Pillow)
- diffusers==0.32.2
- safetensors
- transformers
- peft

Install dependencies:
```bash
# For Latent Color Tools
pip install kornia>=0.6.0 color-matcher>=0.2.0

# For Flux LayerDiffuse
pip install diffusers==0.32.2 safetensors transformers peft
```

## 🚀 Quick Start - Flux LayerDiffuse

### Basic Transparent Image Workflow:

1. **Load Models:**
   - UNETLoader → Flux Model
   - LoraLoaderModelOnly → layerlora.safetensors (Strength: 1.0)
   - DualCLIPLoader → type: "flux"
   - VAELoader → ae.safetensors

2. **Text Encoding:**
   - CLIPTextEncode → "transparent glass bottle"
   - ConditioningZeroOut → for negative

3. **Generation:**
   - EmptySD3LatentImage → 1024x1024
   - KSampler → Standard sampling

4. **Transparent Decoding:**
   - Flux LayerDiffuse Standalone Loader
   - Flux LayerDiffuse Decoder (Simple)

5. **Save/Preview:**
   - Save Transparent Image
   - Preview Transparent Image

### Example Prompts:
- ✅ "transparent glass bottle, elegant"
- ✅ "crystal wine glass, studio lighting"
- ✅ "clear ice cube, frozen water"
- ❌ "transparent car" (too complex)

## Usage

### Color Generator:
1. Add the "Color Generator" node to your workflow
2. Set the desired width and height
3. Select a color from the dropdown
4. Connect the outputs to other nodes as needed

### Load Image Sequence:
1. Add the "📁 Load Image Sequence" node to your workflow
2. Set the folder_path to your image directory
3. Configure sorting method and loop options
4. Each workflow execution will load the next image in sequence
5. Use "📊 Load Image Sequence Info" to monitor progress

**Example workflow:**
```
📁 Load Image Sequence → Image Processing → Save Image
```

### Flux LayerDiffuse:
1. Follow the model installation guide above
2. Use the workflow helper nodes for guidance
3. Check the included documentation files for detailed instructions
4. Use troubleshooter nodes if you encounter issues

## 🎯 Troubleshooting

### Common Issues:

1. **"mat1 and mat2 shapes cannot be multiplied"**
   - Use DualCLIPLoader with type="flux"
   - Use Conditioning Inspector to verify dimensions

2. **"Glass always generated"**
   - This is normal - LayerDiffuse LoRA is trained on glass objects
   - Reduce LoRA strength to 0.5-0.8
   - Use stronger negative prompts: "glass, bottle, wine glass"

3. **"No transparency in output"**
   - Use Save Transparent Image node (not standard Save Image)
   - Check with Transparent Image Info node

4. **Models not found**
   - Check file locations in model folders
   - Use Flux LayerDiffuse Info node to verify

## 📚 Documentation

This package includes comprehensive documentation:
- `COMPLETE_WORKFLOW.md` - Full Flux LayerDiffuse workflow guide
- `GLASS_PROBLEM_SOLUTION.md` - Solutions for glass generation issue
- `TENSOR_DIMENSION_FIX.md` - Fixing dimension errors
- `LOAD_IMAGE_SEQUENCE_GUIDE.md` - Complete guide for image sequence loading
- Various troubleshooting guides

## 🙏 Acknowledgments

Special thanks to:

- **[RedAIGC](https://github.com/RedAIGC)** for developing [Flux-version-LayerDiffuse](https://github.com/RedAIGC/Flux-version-LayerDiffuse) and training the Flux models that make transparent image generation possible

- **[lllyasviel](https://github.com/lllyasviel)** for their groundbreaking work on [LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse) (SD 1.5 version) that laid the foundation for this technology

- **[lum3on](https://github.com/lum3on)** for their invaluable support, inspiration, and encouragement to create these custom nodes

Without their contributions, this project would not have been possible!

