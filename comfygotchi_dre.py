try:
    from custom_nodes.comfygotchi.nodes import ComfyGotchiNode as _ComfyGotchiNode
    COMFYGOTCHI_AVAILABLE = True
except Exception:
    _ComfyGotchiNode = None
    COMFYGOTCHI_AVAILABLE = False

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if _ComfyGotchiNode is not None:
    NODE_CLASS_MAPPINGS["ComfyGotchiNode_DRE"] = _ComfyGotchiNode
    NODE_DISPLAY_NAME_MAPPINGS["ComfyGotchiNode_DRE"] = "ComfyGotchi *DRE"
