from .color_generator_node import ColorGeneratorNode

NODE_CLASS_MAPPINGS = {
    "ColorGeneratorNode": ColorGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGeneratorNode": "Color Generator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
