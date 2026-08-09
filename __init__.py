# Import base nodes with error handling
print("Loading denrakeiw_nodes...")

# Base nodes - always available
try:
    from .color_generator_node import ColorGeneratorNode
    print("✓ ColorGeneratorNode loaded")
except ImportError as e:
    print(f"✗ ColorGeneratorNode failed: {e}")
    ColorGeneratorNode = None

try:
    from .load_image_sequence import LoadImageSequence, LoadImageSequenceInfo
    print("✓ LoadImageSequence loaded")
except ImportError as e:
    print(f"✗ LoadImageSequence failed: {e}")
    LoadImageSequence = None
    LoadImageSequenceInfo = None

try:
    from .latent_colormatch import LatentColorMatch, LatentColorMatchSimple
    print("✓ LatentColorMatch loaded")
except ImportError as e:
    print(f"✗ LatentColorMatch failed: {e}")
    LatentColorMatch = None
    LatentColorMatchSimple = None

try:
    from .latent_adjust import LatentImageAdjust
    print("✓ LatentImageAdjust loaded")
except ImportError as e:
    print(f"✗ LatentImageAdjust failed: {e}")
    LatentImageAdjust = None

try:
    from .multi_image_aspect_ratio_composer import MultiImageAspectRatioComposer
    print("✓ MultiImageAspectRatioComposer loaded")
except ImportError as e:
    print(f"✗ MultiImageAspectRatioComposer failed: {e}")
    MultiImageAspectRatioComposer = None

try:
    from .simple_utf8_save_text import UTF8CaptionSaver
    print("✓ UTF8CaptionSaver loaded")
except ImportError as e:
    print(f"✗ UTF8CaptionSaver failed: {e}")
    UTF8CaptionSaver = None

# Import Universal Latent Upscaler (custom DenRakEiw V2.0 upscaler)
try:
    from .wan_nn_latent_upscaler import WanNNLatentUpscalerNode
    WAN_NN_AVAILABLE = True
    print("✓ Universal Latent Upscaler loaded successfully")
except ImportError as e:
    WAN_NN_AVAILABLE = False
    WanNNLatentUpscalerNode = None
    print(f"⚠ Universal Latent Upscaler not available: {e}")

# Import Universal NN Latent Upscaler (from WAN_NN_Latent_Upscale repo - Ttl based)
# Registered with _DRE suffix to avoid conflicts with the standalone pack.
try:
    from .nn_upscale import UniversalNNLatentUpscale
    UNIVERSAL_NN_UPSCALE_AVAILABLE = True
    print("✓ Universal NN Latent Upscale (WAN_NN_Latent_Upscale) loaded successfully")
except ImportError as e:
    UNIVERSAL_NN_UPSCALE_AVAILABLE = False
    UniversalNNLatentUpscale = None
    print(f"⚠ Universal NN Latent Upscale (WAN_NN_Latent_Upscale) not available: {e}")

# Import Flux LayerDiffuse nodes
try:
    from .flux_layerdiffuse_standalone import FluxLayerDiffuseStandaloneLoader, FluxLayerDiffuseInfo
    from .flux_layerdiffuse_decoder_only import FluxLayerDiffuseDecoderSimple
    from .flux_layerdiffuse_conditioning import FluxLayerDiffuseConditioningFix, FluxLayerDiffuseEmptyConditioning
    from .conditioning_inspector import ConditioningInspector
    from .transparent_image_nodes import SaveTransparentImage, PreviewTransparentImage, TransparentImageInfo

    FLUX_LAYERDIFFUSE_AVAILABLE = True
    print("✓ Flux LayerDiffuse nodes loaded successfully")
except ImportError as e:
    FLUX_LAYERDIFFUSE_AVAILABLE = False
    FluxLayerDiffuseStandaloneLoader = None
    FluxLayerDiffuseInfo = None
    FluxLayerDiffuseDecoderSimple = None
    FluxLayerDiffuseConditioningFix = None
    FluxLayerDiffuseEmptyConditioning = None
    ConditioningInspector = None
    SaveTransparentImage = None
    PreviewTransparentImage = None
    TransparentImageInfo = None
    print(f"⚠ Flux LayerDiffuse nodes not available: {e}")
    print("Install required dependencies: pip install diffusers==0.32.2 safetensors transformers peft")

# Import LLM Deforum Generator
try:
    from .llm_deforum import LLMDeforumGenerator
    LLM_DEFORUM_AVAILABLE = True
    print("✓ LLM Deforum Generator loaded successfully")
except ImportError as e:
    LLM_DEFORUM_AVAILABLE = False
    LLMDeforumGenerator = None
    print(f"⚠ LLM Deforum Generator not available: {e}")

# Import Flux 3 API nodes
try:
    from .flux3_nodes import NODE_CLASS_MAPPINGS as FLUX3_MAPPINGS
    from .flux3_nodes import NODE_DISPLAY_NAME_MAPPINGS as FLUX3_DISPLAY
    FLUX3_AVAILABLE = True
    print("✓ Flux 3 API nodes loaded successfully")
except ImportError as e:
    FLUX3_AVAILABLE = False
    FLUX3_MAPPINGS = {}
    FLUX3_DISPLAY = {}
    print(f"⚠ Flux 3 API nodes not available: {e}")

# Base node mappings - only add nodes that loaded successfully
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if ColorGeneratorNode is not None:
    NODE_CLASS_MAPPINGS["ColorGeneratorNode"] = ColorGeneratorNode
    NODE_DISPLAY_NAME_MAPPINGS["ColorGeneratorNode"] = "Color Generator"

if LoadImageSequence is not None:
    NODE_CLASS_MAPPINGS["LoadImageSequence"] = LoadImageSequence
    NODE_DISPLAY_NAME_MAPPINGS["LoadImageSequence"] = "📁 Load Image Sequence"

if LoadImageSequenceInfo is not None:
    NODE_CLASS_MAPPINGS["LoadImageSequenceInfo"] = LoadImageSequenceInfo
    NODE_DISPLAY_NAME_MAPPINGS["LoadImageSequenceInfo"] = "📊 Load Image Sequence Info"

if LatentColorMatch is not None:
    NODE_CLASS_MAPPINGS["LatentColorMatch_DRE"] = LatentColorMatch
    NODE_DISPLAY_NAME_MAPPINGS["LatentColorMatch_DRE"] = "🎨 Latent Color Match *DRE"

if LatentColorMatchSimple is not None:
    NODE_CLASS_MAPPINGS["LatentColorMatchSimple_DRE"] = LatentColorMatchSimple
    NODE_DISPLAY_NAME_MAPPINGS["LatentColorMatchSimple_DRE"] = "🎨 Latent Color Match (Simple) *DRE"

if LatentImageAdjust is not None:
    NODE_CLASS_MAPPINGS["LatentImageAdjust_DRE"] = LatentImageAdjust
    NODE_DISPLAY_NAME_MAPPINGS["LatentImageAdjust_DRE"] = "🎛️ Latent Image Adjust *DRE"

if MultiImageAspectRatioComposer is not None:
    NODE_CLASS_MAPPINGS["MultiImageAspectRatioComposer"] = MultiImageAspectRatioComposer
    NODE_DISPLAY_NAME_MAPPINGS["MultiImageAspectRatioComposer"] = "🖼️ Multi-Image Aspect Ratio Composer"

if UTF8CaptionSaver is not None:
    NODE_CLASS_MAPPINGS["UTF8CaptionSaver"] = UTF8CaptionSaver
    NODE_DISPLAY_NAME_MAPPINGS["UTF8CaptionSaver"] = "📝 UTF-8 Caption Saver"

# Add Universal Latent Upscaler if available (custom DenRakEiw V2.0)
if WAN_NN_AVAILABLE and WanNNLatentUpscalerNode is not None:
    NODE_CLASS_MAPPINGS["WanNNLatentUpscaler"] = WanNNLatentUpscalerNode
    NODE_DISPLAY_NAME_MAPPINGS["WanNNLatentUpscaler"] = "Universal Latent Upscaler"

# Add Universal NN Latent Upscale (from WAN_NN_Latent_Upscale repo) if available
# _DRE suffix avoids conflict with the standalone WAN_NN_Latent_Upscale pack.
if UNIVERSAL_NN_UPSCALE_AVAILABLE and UniversalNNLatentUpscale is not None:
    NODE_CLASS_MAPPINGS["UniversalNNLatentUpscale_DRE"] = UniversalNNLatentUpscale
    NODE_DISPLAY_NAME_MAPPINGS["UniversalNNLatentUpscale_DRE"] = "🚀 Universal NN Latent Upscale *DRE"

# Add Flux LayerDiffuse nodes if available
if FLUX_LAYERDIFFUSE_AVAILABLE:
    if FluxLayerDiffuseStandaloneLoader is not None:
        NODE_CLASS_MAPPINGS["FluxLayerDiffuseStandaloneLoader"] = FluxLayerDiffuseStandaloneLoader
        NODE_DISPLAY_NAME_MAPPINGS["FluxLayerDiffuseStandaloneLoader"] = "🔧 Flux LayerDiffuse Standalone Loader"

    if FluxLayerDiffuseInfo is not None:
        NODE_CLASS_MAPPINGS["FluxLayerDiffuseInfo"] = FluxLayerDiffuseInfo
        NODE_DISPLAY_NAME_MAPPINGS["FluxLayerDiffuseInfo"] = "ℹ️ Flux LayerDiffuse Info"

    if FluxLayerDiffuseDecoderSimple is not None:
        NODE_CLASS_MAPPINGS["FluxLayerDiffuseDecoderSimple"] = FluxLayerDiffuseDecoderSimple
        NODE_DISPLAY_NAME_MAPPINGS["FluxLayerDiffuseDecoderSimple"] = "🔍 Flux LayerDiffuse Decoder (Simple)"

    if FluxLayerDiffuseConditioningFix is not None:
        NODE_CLASS_MAPPINGS["FluxLayerDiffuseConditioningFix"] = FluxLayerDiffuseConditioningFix
        NODE_DISPLAY_NAME_MAPPINGS["FluxLayerDiffuseConditioningFix"] = "🔧 Flux LayerDiffuse Conditioning Fix"

    if FluxLayerDiffuseEmptyConditioning is not None:
        NODE_CLASS_MAPPINGS["FluxLayerDiffuseEmptyConditioning"] = FluxLayerDiffuseEmptyConditioning
        NODE_DISPLAY_NAME_MAPPINGS["FluxLayerDiffuseEmptyConditioning"] = "⭕ Flux LayerDiffuse Empty Conditioning"

    if ConditioningInspector is not None:
        NODE_CLASS_MAPPINGS["ConditioningInspector"] = ConditioningInspector
        NODE_DISPLAY_NAME_MAPPINGS["ConditioningInspector"] = "🔍 Conditioning Inspector"

    if SaveTransparentImage is not None:
        NODE_CLASS_MAPPINGS["SaveTransparentImage"] = SaveTransparentImage
        NODE_DISPLAY_NAME_MAPPINGS["SaveTransparentImage"] = "💾 Save Transparent Image"

    if PreviewTransparentImage is not None:
        NODE_CLASS_MAPPINGS["PreviewTransparentImage"] = PreviewTransparentImage
        NODE_DISPLAY_NAME_MAPPINGS["PreviewTransparentImage"] = "👁️ Preview Transparent Image"

    if TransparentImageInfo is not None:
        NODE_CLASS_MAPPINGS["TransparentImageInfo"] = TransparentImageInfo
        NODE_DISPLAY_NAME_MAPPINGS["TransparentImageInfo"] = "📊 Transparent Image Info"

# Add LLM Deforum Generator if available
if LLM_DEFORUM_AVAILABLE and LLMDeforumGenerator is not None:
    NODE_CLASS_MAPPINGS["LLMDeforumGenerator"] = LLMDeforumGenerator
    NODE_DISPLAY_NAME_MAPPINGS["LLMDeforumGenerator"] = "🤖 LLM Deforum (WIP)"

# Add Flux 3 API nodes if available
if FLUX3_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(FLUX3_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(FLUX3_DISPLAY)

print(f"✓ denrakeiw_nodes loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully")

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
