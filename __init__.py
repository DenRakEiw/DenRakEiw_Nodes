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

# Import Universal Latent Upscaler
try:
    from .wan_nn_latent_upscaler import WanNNLatentUpscalerNode
    WAN_NN_AVAILABLE = True
    print("✓ Universal Latent Upscaler loaded successfully")
except ImportError as e:
    WAN_NN_AVAILABLE = False
    WanNNLatentUpscalerNode = None
    print(f"⚠ Universal Latent Upscaler not available: {e}")

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
    NODE_CLASS_MAPPINGS["LatentColorMatch"] = LatentColorMatch
    NODE_DISPLAY_NAME_MAPPINGS["LatentColorMatch"] = "🎨 Latent Color Match"

if LatentColorMatchSimple is not None:
    NODE_CLASS_MAPPINGS["LatentColorMatchSimple"] = LatentColorMatchSimple
    NODE_DISPLAY_NAME_MAPPINGS["LatentColorMatchSimple"] = "🎨 Latent Color Match (Simple)"

if LatentImageAdjust is not None:
    NODE_CLASS_MAPPINGS["LatentImageAdjust"] = LatentImageAdjust
    NODE_DISPLAY_NAME_MAPPINGS["LatentImageAdjust"] = "🎛️ Latent Image Adjust"

if MultiImageAspectRatioComposer is not None:
    NODE_CLASS_MAPPINGS["MultiImageAspectRatioComposer"] = MultiImageAspectRatioComposer
    NODE_DISPLAY_NAME_MAPPINGS["MultiImageAspectRatioComposer"] = "🖼️ Multi-Image Aspect Ratio Composer"

# Add Universal Latent Upscaler if available
if WAN_NN_AVAILABLE and WanNNLatentUpscalerNode is not None:
    NODE_CLASS_MAPPINGS["WanNNLatentUpscaler"] = WanNNLatentUpscalerNode
    NODE_DISPLAY_NAME_MAPPINGS["WanNNLatentUpscaler"] = "Universal Latent Upscaler"

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

print(f"✓ denrakeiw_nodes loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully")

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
