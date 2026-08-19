import os
import json
import threading
from datetime import datetime, timezone

from .state import GotchiState, CONFIG, DEFAULT_STATE

STATE_FILE = os.path.join(os.path.dirname(__file__), "state_dre.json")
_lock = threading.Lock()
_state = None
_state_mtime = None

def _disk_mtime():
    try:
        return os.path.getmtime(STATE_FILE)
    except OSError:
        return None

def _get_state():
    global _state, _state_mtime
    mtime = _disk_mtime()
    if _state is None:
        _state = GotchiState.load(STATE_FILE)
        _state_mtime = mtime
    elif mtime is not None and _state_mtime is not None and mtime != _state_mtime:
        _state = GotchiState.load(STATE_FILE)
        _state_mtime = mtime
    return _state

def _reset_state():
    global _state, _state_mtime
    _state = GotchiState()
    _state.save(STATE_FILE)
    _state_mtime = _disk_mtime()

def _save_state():
    global _state_mtime
    s = _get_state()
    s.save(STATE_FILE)
    _state_mtime = _disk_mtime()

def _now():
    return datetime.now(timezone.utc)

def _apply_decay_on_read():
    s = _get_state()
    if s.last_decay_at and s.stage not in ("egg", "ghost"):
        try:
            last = datetime.fromisoformat(s.last_decay_at)
            elapsed = (_now() - last).total_seconds() / 60.0
            if elapsed > 0:
                capped = min(elapsed, CONFIG.CATCHUP_DECAY_MAX_MIN)
                s.apply_tick(capped)
            else:
                s.last_decay_at = _now().isoformat()
        except (ValueError, TypeError):
            s.last_decay_at = _now().isoformat()
    elif not s.last_decay_at:
        s.last_decay_at = _now().isoformat()
    _save_state()
    return s

def init_server(server_instance):
    from server import PromptServer
    from aiohttp import web

    @server_instance.routes.get("/comfygotchi_dre/state")
    async def get_state(request):
        with _lock:
            s = _apply_decay_on_read()
            return web.json_response(s.to_dict())

    @server_instance.routes.post("/comfygotchi_dre/event")
    async def post_event(request):
        body = await request.json()
        event_type = body.get("type", "")
        caption = body.get("caption", "")
        with _lock:
            s = _get_state()
            if event_type == "feed":
                s.apply_feed()
            elif event_type == "comment":
                if caption:
                    s.comment_history.append(caption)
                    if len(s.comment_history) > 50:
                        s.comment_history = s.comment_history[-50:]
                s.last_comment_qwen = bool(body.get("qwen", False))
            elif event_type == "love":
                s.apply_love()
            elif event_type == "tick":
                elapsed = body.get("elapsed_minutes", 1.0)
                s.apply_tick(elapsed)
                if s.stage == "ghost" and s.died_at:
                    try:
                        died = datetime.fromisoformat(s.died_at)
                        ghost_min = (_now() - died).total_seconds() / 60.0
                        s.check_reincarnation(ghost_min, 0)
                    except (ValueError, TypeError):
                        pass
            elif event_type == "egg_caption":
                if hasattr(s, "egg_captions") and caption:
                    s.egg_captions.append(caption)
            elif event_type == "set_variant":
                v = body.get("variant", "blob")
                p = body.get("personality", "")
                s.variant = v
                s.personality = p
                s.variant_determined = True
            elif event_type == "play":
                s.apply_play()
                s.comment_history.append("Yay! Let's play!")
                if len(s.comment_history) > 50: s.comment_history = s.comment_history[-50:]
            elif event_type == "clean":
                s.apply_clean()
                s.comment_history.append("All clean now!")
                if len(s.comment_history) > 50: s.comment_history = s.comment_history[-50:]
            elif event_type == "medicine":
                s.apply_medicine()
                s.comment_history.append("Ugh... but I feel better.")
                if len(s.comment_history) > 50: s.comment_history = s.comment_history[-50:]
            s.last_event_at = _now().isoformat()
            _save_state()
            return web.json_response(s.to_dict())

    @server_instance.routes.post("/comfygotchi_dre/save")
    async def post_save(request):
        with _lock:
            _save_state()
            return web.json_response({"ok": True})

    @server_instance.routes.post("/comfygotchi_dre/reset")
    async def post_reset(request):
        with _lock:
            _reset_state()
            return web.json_response(_get_state().to_dict())

    @server_instance.routes.get("/comfygotchi_dre/config")
    async def get_config(request):
        return web.json_response({
            "HATCH_THRESHOLD": CONFIG.HATCH_THRESHOLD,
            "EVOLUTION_THRESHOLD": CONFIG.EVOLUTION_THRESHOLD,
            "DELTA_FEED": CONFIG.DELTA_FEED,
            "DELTA_LOVE": CONFIG.DELTA_LOVE,
            "DELTA_HUNGER_PER_MIN": CONFIG.DELTA_HUNGER_PER_MIN,
        })

try:
    from server import PromptServer
    SERVER = PromptServer.instance
    init_server(SERVER)
except Exception as e:
    print(f"[ComfyGotchi_DRE] Could not register server routes: {e}")
