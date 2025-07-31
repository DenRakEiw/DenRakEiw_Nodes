from .color_generator_node import ColorGeneratorNode

# Import Flux LayerDiffuse nodes
try:
    from .flux_layerdiffuse_standalone import FluxLayerDiffuseStandaloneLoader, FluxLayerDiffuseInfo
    from .flux_layerdiffuse_decoder_only import FluxLayerDiffuseDecoderSimple, FluxLayerDiffuseWorkflowHelper
    from .flux_layerdiffuse_conditioning import FluxLayerDiffuseConditioningFix, FluxLayerDiffuseEmptyConditioning, FluxLayerDiffuseTroubleshooter
    from .conditioning_inspector import ConditioningInspector
    from .transparent_image_nodes import SaveTransparentImage, PreviewTransparentImage, TransparentImageInfo

    FLUX_LAYERDIFFUSE_AVAILABLE = True
    print("✓ Flux LayerDiffuse nodes loaded successfully")
except ImportError as e:
    FLUX_LAYERDIFFUSE_AVAILABLE = False
    print(f"⚠ Flux LayerDiffuse nodes not available: {e}")
    print("Install required dependencies: pip install diffusers==0.32.2 safetensors transformers peft")

# Base node mappings
NODE_CLASS_MAPPINGS = {
    "ColorGeneratorNode": ColorGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGeneratorNode": "Color Generator",
}

# Add Flux LayerDiffuse nodes if available
if FLUX_LAYERDIFFUSE_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        # Model loading
        "FluxLayerDiffuseStandaloneLoader": FluxLayerDiffuseStandaloneLoader,
        "FluxLayerDiffuseInfo": FluxLayerDiffuseInfo,

        # Decoding and workflow help
        "FluxLayerDiffuseDecoderSimple": FluxLayerDiffuseDecoderSimple,
        "FluxLayerDiffuseWorkflowHelper": FluxLayerDiffuseWorkflowHelper,

        # Conditioning and troubleshooting
        "FluxLayerDiffuseConditioningFix": FluxLayerDiffuseConditioningFix,
        "FluxLayerDiffuseEmptyConditioning": FluxLayerDiffuseEmptyConditioning,
        "FluxLayerDiffuseTroubleshooter": FluxLayerDiffuseTroubleshooter,
        "ConditioningInspector": ConditioningInspector,

        # Transparent image handling
        "SaveTransparentImage": SaveTransparentImage,
        "PreviewTransparentImage": PreviewTransparentImage,
        "TransparentImageInfo": TransparentImageInfo,
    })

    NODE_DISPLAY_NAME_MAPPINGS.update({
        # Model loading
        "FluxLayerDiffuseStandaloneLoader": "🔧 Flux LayerDiffuse Standalone Loader",
        "FluxLayerDiffuseInfo": "ℹ️ Flux LayerDiffuse Info",

        # Decoding and workflow help
        "FluxLayerDiffuseDecoderSimple": "🔍 Flux LayerDiffuse Decoder (Simple)",
        "FluxLayerDiffuseWorkflowHelper": "📖 Flux LayerDiffuse Workflow Helper",

        # Conditioning and troubleshooting
        "FluxLayerDiffuseConditioningFix": "🔧 Flux LayerDiffuse Conditioning Fix",
        "FluxLayerDiffuseEmptyConditioning": "⭕ Flux LayerDiffuse Empty Conditioning",
        "FluxLayerDiffuseTroubleshooter": "🩺 Flux LayerDiffuse Troubleshooter",
        "ConditioningInspector": "🔍 Conditioning Inspector",

        # Transparent image handling
        "SaveTransparentImage": "💾 Save Transparent Image",
        "PreviewTransparentImage": "👁️ Preview Transparent Image",
        "TransparentImageInfo": "📊 Transparent Image Info",
    })

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
