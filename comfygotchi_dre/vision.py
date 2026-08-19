import os
import glob
import numpy as np

_QWEN_STATE = {
    "model": None,
    "processor": None,
    "tokenizer": None,
    "current_path": None,
}

def detect_qwen_models():
    try:
        import folder_paths
        llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    except Exception:
        llm_paths = []
    if not llm_paths:
        llm_paths = [os.path.join(os.path.dirname(__file__), "..", "..", "models", "LLM")]
    
    models = ["none (rule-based)"]
    for base in llm_paths:
        qwen_dir = os.path.join(base, "Qwen-VL")
        if not os.path.isdir(qwen_dir):
            continue
        for entry in os.listdir(qwen_dir):
            full = os.path.join(qwen_dir, entry)
            if not os.path.isdir(full):
                continue
            has_config = os.path.exists(os.path.join(full, "config.json"))
            has_weights = bool(glob.glob(os.path.join(full, "*.safetensors")) or glob.glob(os.path.join(full, "*.bin")))
            if has_config and has_weights:
                models.append(entry)
    return models

def _tensor_to_pil(tensor):
    if tensor is None:
        return None
    from PIL import Image
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    arr = tensor
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _load_qwen(model_path):
    if _QWEN_STATE["current_path"] == model_path and _QWEN_STATE["model"] is not None:
        return
    _unload_qwen()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[ComfyGotchi_DRE] Loading Qwen-VLM from {model_path} on {device}...")
    _QWEN_STATE["model"] = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    _QWEN_STATE["processor"] = AutoProcessor.from_pretrained(model_path)
    _QWEN_STATE["tokenizer"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _QWEN_STATE["current_path"] = model_path
    print("[ComfyGotchi_DRE] Qwen-VLM loaded.")

def _unload_qwen():
    if _QWEN_STATE["model"] is not None:
        try:
            _QWEN_STATE["model"] = _QWEN_STATE["model"].cpu()
        except Exception:
            pass
    _QWEN_STATE["model"] = None
    _QWEN_STATE["processor"] = None
    _QWEN_STATE["tokenizer"] = None
    _QWEN_STATE["current_path"] = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

def _get_model_path(model_name):
    try:
        import folder_paths
        llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    except Exception:
        llm_paths = []
    if not llm_paths:
        llm_paths = [os.path.join(os.path.dirname(__file__), "..", "..", "models", "LLM")]
    for base in llm_paths:
        candidate = os.path.join(base, "Qwen-VL", model_name)
        if os.path.isdir(candidate):
            return candidate
    return None

def _rule_based_caption(image_tensor):
    if image_tensor is None:
        return "something"
    arr = image_tensor
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    mean_brightness = float(arr.mean())
    r, g, b = float(arr[..., 0].mean()), float(arr[..., 1].mean()), float(arr[..., 2].mean())
    dominant = ["red", "green", "blue"][np.argmax([r, g, b])]
    if mean_brightness < 0.2:
        return f"a dark {dominant} image"
    if mean_brightness > 0.8:
        return f"a bright {dominant} image"
    return f"a {dominant} image"

def _safe_pad_token_id(tokenizer):
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is None:
        pid = getattr(tokenizer, "eos_token_id", None)
    return pid

def _qwen_generate(pil_image, prompt_text, max_tokens=128, retries=2):
    import torch
    model = _QWEN_STATE["model"]
    processor = _QWEN_STATE["processor"]
    tokenizer = _QWEN_STATE["tokenizer"]
    if model is None or processor is None or tokenizer is None:
        raise RuntimeError("Qwen model not loaded")
    pad_token_id = _safe_pad_token_id(tokenizer)

    conversation = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt_text}
    ]}]
    chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=[pil_image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    last_text = ""
    for attempt in range(retries + 1):
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6 if attempt == 0 else 0.3,
            top_p=0.9,
        )
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        output = model.generate(**gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
        if text:
            return text
        last_text = text
        print(f"[ComfyGotchi_DRE] Qwen returned empty caption, retry {attempt + 1}/{retries + 1}")
    return last_text

def _qwen_text_only(prompt_text, max_tokens=256):
    import torch
    model = _QWEN_STATE["model"]
    processor = _QWEN_STATE["processor"]
    tokenizer = _QWEN_STATE["tokenizer"]
    if model is None or processor is None or tokenizer is None:
        raise RuntimeError("Qwen model not loaded")
    pad_token_id = _safe_pad_token_id(tokenizer)

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    output = model.generate(**gen_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
    return text.strip()

def caption_image(image_tensor, model_name="none (rule-based)", keep_model_loaded=True):
    if model_name == "none (rule-based)" or model_name is None:
        return _rule_based_caption(image_tensor)
    
    model_path = _get_model_path(model_name)
    if model_path is None:
        print(f"[ComfyGotchi_DRE] Qwen model '{model_name}' not found, falling back")
        return _rule_based_caption(image_tensor)
    
    try:
        _load_qwen(model_path)
        pil_image = _tensor_to_pil(image_tensor)
        caption = _qwen_generate(pil_image, "Describe this image in one short sentence.", max_tokens=64)
        if not keep_model_loaded:
            _unload_qwen()
        if not caption:
            print("[ComfyGotchi_DRE] Qwen returned empty caption after retries, using rule-based fallback")
            return _rule_based_caption(image_tensor)
        return caption
    except Exception as e:
        print(f"[ComfyGotchi_DRE] Qwen caption failed: {e}, falling back")
        try:
            _unload_qwen()
        except Exception:
            pass
        return _rule_based_caption(image_tensor)

_VARIANT_KEYWORDS = {
    "cat": ["cat", "kitten", "kitty", "feline", "tabby", "calico", "siamese", "ginger cat", "orange cat"],
    "dog": ["dog", "puppy", "canine", "retriever", "poodle", "beagle", "husky", "corgi", "shepherd", "terrier", "bulldog", "shiba", "labrador"],
    "monster": ["monster", "demon", "creature", "beast", "alien monster", "ghoul", "ogre", "troll", "horror"],
    "dragon": ["dragon", "wyvern", "lizard", "reptile", "serpent", "dino", "dinosaur", "scales", "winged beast"],
    "robot": ["robot", "android", "cyborg", "machine", "mechanical", "android", "droid", "bot", "cyber", "metallic"],
    "phantom": ["ghost", "phantom", "spirit", "specter", "wraith", "shadowy", "ethereal", "apparition", "haunted"],
    "alien": ["alien", "extraterrestrial", "ufo", "martian", "space creature", "green creature", "martian"],
    "bunny": ["bunny", "rabbit", "hare", "lagomorph", "bunnies", "easter bunny"],
    "penguin": ["penguin", "bird", "arctic", "ice", "flightless", "puffin", "antarctic"],
}

def _detect_variant_by_keywords(captions):
    scores = {v: 0 for v in _VARIANT_KEYWORDS}
    for cap in captions:
        low = (cap or "").lower()
        for variant, keywords in _VARIANT_KEYWORDS.items():
            for kw in keywords:
                if kw in low:
                    scores[variant] += 1
                    break  # one match per caption per variant
    max_score = max(scores.values())
    if max_score == 0:
        return None, 0
    winners = [v for v, s in scores.items() if s == max_score]
    if len(winners) > 1:
        return None, 0  # ambiguous tie
    return winners[0], max_score

def _detect_personality_by_keywords(captions):
    text = " ".join(captions).lower()
    personality_tags = []
    if any(w in text for w in ["dark", "night", "shadow", "noir", "moody", "gothic", "black"]):
        personality_tags.append("dark")
    if any(w in text for w in ["bright", "sunny", "colorful", "vibrant", "cheerful", "rainbow"]):
        personality_tags.append("bright")
    if any(w in text for w in ["nature", "forest", "tree", "grass", "garden", "flower", "mountain", "organic"]):
        personality_tags.append("nature")
    if any(w in text for w in ["tech", "robot", "circuit", "cyber", "metal", "steel", "digital", "sci-fi", "futuristic"]):
        personality_tags.append("tech")
    if any(w in text for w in ["cute", "soft", "fluffy", "kawaii", "adorable", "cozy"]):
        personality_tags.append("cute")
    if any(w in text for w in ["horror", "scary", "creepy", "blood", "skull", "dark", "evil"]):
        personality_tags.append("moody")
    return " ".join(personality_tags[:3]) if personality_tags else ""

def determine_variant(egg_captions, model_name="none (rule-based)", keep_model_loaded=True):
    variants = ["blob", "cat", "dog", "monster", "dragon", "robot", "phantom", "alien", "bunny", "penguin"]

    if not egg_captions:
        return "blob", ""

    # PRIMARY: keyword-based detection — reliable, deterministic
    kw_variant, kw_score = _detect_variant_by_keywords(egg_captions)
    kw_personality = _detect_personality_by_keywords(egg_captions)

    if kw_variant and kw_score >= 3:
        # Strong keyword signal — use it directly
        print(f"[ComfyGotchi_DRE] Variant from keywords: {kw_variant} (score={kw_score}), personality='{kw_personality}'")
        return kw_variant, kw_personality

    # FALLBACK: ask Qwen if available and keyword signal is weak
    if model_name == "none (rule-based)" or model_name is None:
        if kw_variant:
            print(f"[ComfyGotchi_DRE] Weak keyword signal ({kw_variant} score={kw_score}), using it anyway (no Qwen)")
            return kw_variant, kw_personality
        return "blob", kw_personality

    model_path = _get_model_path(model_name)
    if model_path is None:
        return kw_variant or "blob", kw_personality

    try:
        _load_qwen(model_path)
        captions_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(egg_captions))
        prompt = f"""You are deciding a tamagotchi creature's identity. Based on these 10 image descriptions, choose:
1. A variant from: blob, cat, dog, monster, dragon, robot, phantom, alien, bunny, penguin
2. A personality: 2-3 keywords describing the user's aesthetic

Image descriptions:
{captions_text}

Respond ONLY as JSON: {{"variant": "cat", "personality": "dark moody cinematic"}}"""

        response = _qwen_text_only(prompt, max_tokens=128)
        if not keep_model_loaded:
            _unload_qwen()

        import json as _json
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            result = _json.loads(response)
            variant = result.get("variant", "blob").lower().strip()
            if variant not in variants:
                variant = kw_variant or "blob"
            personality = result.get("personality", "").strip()
            if not personality:
                personality = kw_personality
            # Prefer keyword variant if Qwen disagrees and keyword score was decent
            if kw_variant and kw_score >= 2 and variant != kw_variant:
                print(f"[ComfyGotchi_DRE] Qwen said {variant} but keywords say {kw_variant} (score={kw_score}), trusting keywords")
                variant = kw_variant
            print(f"[ComfyGotchi_DRE] Variant from Qwen: {variant}, personality='{personality}'")
            return variant, personality
        except (_json.JSONDecodeError, KeyError):
            for v in variants:
                if v in response.lower():
                    return v, kw_personality
            return kw_variant or "blob", kw_personality
    except Exception as e:
        print(f"[ComfyGotchi_DRE] Variant determination failed: {e}")
        return kw_variant or "blob", kw_personality
