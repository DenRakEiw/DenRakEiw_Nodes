# Denrakeiw Nodes

Comprehensive ComfyUI custom nodes for **image generation**, **latent space manipulation**, **Flux LayerDiffuse transparency**, and **multi-image composition**.

## 🎨 Core Nodes

### Color Generator
Generates solid color images with 130+ predefined colors.
- **Input**: Width/height (1-8192px), color selection
- **Output**: Image, color name, hex code

### 📁 Load Image Sequence
Sequential image loading from folders with state management.
- **Features**: Auto-increment, sorting options, loop mode, error handling
- **Output**: Current image, filename, index, total count
- **Advantage**: Low memory usage vs batch loading

### 🖼️ Multi-Image Aspect Ratio Composer
Advanced multi-image composition with intelligent layouts.

**🎛️ Dynamic Controls:**
- **Input Count**: 1-8 images with dynamic UI updates
- **Aspect Ratios**: 1:1, 4:3, 16:9, 9:16, 21:9, 3:2, 5:4, 2:1, etc.
- **Megapixels**: 0.5-32 MP with auto-calculation
- **Layouts**: Horizontal, Vertical, Smart Grid, Classic Grid

**🎭 Face Detection:**
- **Haar Cascade**: Fast CPU-based detection
- **DNN Face**: Advanced deep learning detection
- **Smart Cropping**: Centers images around detected faces
- **Confidence**: Adjustable detection threshold (1.1-3.0)

**🎯 Smart Grid Algorithm:**
Optimizes layout based on target aspect ratio:
- **8 images + 16:9**: [4,4] layout (2 rows)
- **8 images + 1:1**: [3,3,2] layout (3 rows)
- **8 images + 9:16**: [2,2,2,2] layout (4 rows)

---

## 🎨 Latent Space Tools

> ℹ️ The latent nodes below are also available as standalone packs
> ([Latent_Nodes](https://github.com/DenRakEiw/Latent_Nodes) and
> [WAN_NN_Latent_Upscale](https://github.com/DenRakEiw/WAN_NN_Latent_Upscale)).
> To allow both packs to coexist in one ComfyUI install without node-ID conflicts,
> the integrated versions here use an internal `_DRE` suffix and a `*DRE` tag in
> the display name (e.g. `🎨 Latent Color Match *DRE`).

### 🎨 Latent Color Match *DRE
Advanced color matching directly in latent space - **10x faster** than image-based methods.
- **Methods**: LAB, YCbCr, LUV, YUV, XYZ, RGB (kornia) + advanced algorithms (color-matcher)
- **Benefits**: No VAE encoding/decoding, 50% less VRAM, no quality loss
- **Controls**: Factor strength (0.0-3.0), batch processing, GPU acceleration

### 🎛️ Latent Image Adjust *DRE
Complete adjustment suite in latent space with kornia integration.
- **Controls**: Brightness (-1.0 to 1.0), Contrast (0.0 to 3.0), Hue (±180°), Saturation (0.0-3.0), Sharpness (0.0-3.0)
- **Features**: Direct latent manipulation, memory efficient, anti-aliasing

### 🚀 Universal NN Latent Upscale *DRE
Neural network-based latent upscaling (from the WAN_NN_Latent_Upscale repo, based on
[Ttl's ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale)).
- **Models**: SD1.5, SDXL, Flux, and Wan2.2 (bundled `.pt` files in `models/`)
- **Scale**: Configurable 1.0x–2.0x, automatic model detection/loading
- **Performance**: Direct latent processing without VAE decode/encode cycles

### Universal Latent Upscaler
Neural network-based latent upscaling (custom DenRakEiw V2.0 auto-detecting 4ch/16ch).
- **Models**: Supports multiple architectures with automatic detection
- **Performance**: Direct latent processing without VAE decode/encode cycles

---

## 🔥 Flux LayerDiffuse (Transparent Images)

Complete transparent image generation system for Flux models.

### Core Nodes:
- **🔧 Standalone Loader**: Loads TransparentVAE.pth model
- **🔍 Decoder (Simple)**: Converts latents to transparent RGBA images
- **💾 Save Transparent**: Saves PNG with proper alpha channel
- **👁️ Preview Transparent**: Multi-background preview (checkerboard/white/black)
- **📊 Transparent Info**: Alpha channel analysis and statistics

### Utility Nodes:
- **🔍 Conditioning Inspector**: Verifies Flux tensor compatibility
- **🔧 Conditioning Fix**: Converts SD to Flux format (if needed)
- **⭕ Empty Conditioning**: Proper negative conditioning for Flux
- **ℹ️ Info**: Model requirements and download links

### Quick Workflow:
1. **Models**: UNETLoader (Flux) + LoraLoader (layerlora.safetensors, strength 1.0)
2. **Text**: DualCLIPLoader (type="flux") + CLIPTextEncode
3. **Generate**: EmptySD3LatentImage + KSampler
4. **Transparent**: Standalone Loader + Decoder + Save/Preview

**Example Prompts**: "transparent glass bottle", "crystal wine glass", "clear ice cube"

---

## 🎬 Flux 3 API (BFL)

Two nodes for the BFL Flux 3 video API and an LLM-powered prompt generator.

### 🎬 Flux 3 Video (API)
Generates video via `POST https://api.bfl.ai/v1/flux-3-video` with synchronized audio.
Four modes: `t2v` (text→video), `i2v` (image→video with keyframes),
`v2v` (video continuation), `draft_enhance` (final-render a prior draft).
Up to full-HD, 5–20 seconds. Async with retries, backoff, and parallel execution.
([Docs](https://docs.bfl.ai/flux_3/flux3_video))

### 🎬 Flux 3 Openrouter Prompt
Turns a plain-words idea into a complete, structured FLUX 3 prompt using an
[OpenRouter](https://openrouter.ai) LLM. The model dropdown is filled **live**
from the OpenRouter API. Optional reference images for vision-capable models.
Output is just the prompt — no variants, no metadata — ready to paste into the
Video node's `prompt` input.

**Skills:** Two prompting skills are pre-installed in the `skills/` folder:
- **Flux3Director** — full structured prompting skill (all modes, timing, camera vocabulary, tag system)
- **Flux3Director4Discord** — compressed for the FLUX3 Discord bot's 2,000-character limit

Drop your own `.md` files into `skills/` and refresh ComfyUI — they appear in the dropdown automatically.

**Setup:** Add to `.env`:
```
BFL_API_KEY=bfl_...
OPENROUTER_API_KEY=sk-or-...
```

---

## 📁 Installation & Requirements

### Installation:
1. Clone to `ComfyUI/custom_nodes/denrakeiw_nodes/`
2. Install dependencies: `pip install -r requirements.txt`
3. Restart ComfyUI

### Required Models (Flux LayerDiffuse):
- **Flux**: `flux1-dev-fp8.safetensors` → `models/diffusion_models/`
- **LoRA**: `layerlora.safetensors` → `models/loras/` (included)
- **TransparentVAE**: `TransparentVAE.pth` → `models/vae/` (included)
- **CLIP**: `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors` → `models/clip/`
- **VAE**: `ae.safetensors` → `models/vae/`

Download from: [Hugging Face - Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev), [LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

### Dependencies:
```bash
pip install kornia>=0.6.0 color-matcher>=0.2.0 diffusers==0.32.2 safetensors transformers peft opencv-python
```

## 🚀 Quick Usage

### Multi-Image Composer:
1. Set **Input Count** (1-8) → Click **Update Inputs**
2. Choose **Aspect Ratio** and **Megapixels**
3. Select **Layout**: Smart Grid (recommended) or Horizontal/Vertical
4. Enable **Face Detection** for portraits (Haar Cascade/DNN)
5. Connect images → Execute

### Latent Color Matching:
```
Image → VAE Encode → Latent Color Match → VAE Decode → Save
```

### Transparent Images (Flux):
```
UNETLoader + LoraLoader (layerlora.safetensors) → KSampler →
Flux LayerDiffuse Decoder → Save Transparent Image
```

## 🎯 Troubleshooting

**Flux LayerDiffuse:**
- **Tensor errors**: Use DualCLIPLoader type="flux" + Conditioning Inspector
- **Glass generation**: Normal behavior - reduce LoRA strength to 0.5-0.8
- **No transparency**: Use Save Transparent Image (not standard Save Image)

**Multi-Image Composer:**
- **Missing images**: Check all inputs connected + click "Update Inputs"
- **Face detection slow**: Disable for non-portrait images
- **Memory issues**: Reduce megapixel setting

## 🙏 Acknowledgments

Thanks to **[RedAIGC](https://github.com/RedAIGC)** (Flux LayerDiffuse), **[lllyasviel](https://github.com/lllyasviel)** (LayerDiffuse foundation), and **[lum3on](https://github.com/lum3on)** (support & inspiration).

