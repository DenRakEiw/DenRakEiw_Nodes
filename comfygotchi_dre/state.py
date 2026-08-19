from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import tempfile

@dataclass
class TunableConfig:
    HATCH_THRESHOLD: int = 10
    N_FEEDS_GROWUP: int = 3
    EVOLUTION_THRESHOLD: int = 50
    DELTA_FEED: float = 15.0
    DELTA_LOVE: float = 10.0
    DELTA_HUNGER_PER_MIN: float = 0.2
    DELTA_HAPPINESS_DECAY_PER_MIN: float = 0.1
    DELTA_BOREDOM_PER_MIN: float = 0.15
    DELTA_POOP_CHANCE: float = 0.08
    DELTA_HYGIENE_DECAY_PER_MIN: float = 0.05
    DELTA_SICKNESS_FROM_POOP: float = 0.3
    DELTA_SICKNESS_FROM_BOREDOM: float = 0.15
    DELTA_SICKNESS_FROM_HUNGER: float = 0.25
    DELTA_SICKNESS_FROM_AGE: float = 0.1
    SICKNESS_DEATH_THRESHOLD: float = 100.0
    AGE_MAX_MINUTES: int = 10080
    AGE_MIN_MINUTES: int = 1440
    T_GHOST_MIN: int = 2
    GHOST_EVENT_THRESHOLD: int = 3
    TICK_TIMEOUT_SEC: int = 120
    CATCHUP_DECAY_MAX_MIN: int = 30
    POOP_MAX: int = 5
    PLAY_HAPPINESS_BOOST: float = 15.0
    CLEAN_HYGIENE_BOOST: float = 80.0
    MEDICINE_SICKNESS_RESET: float = 0.0

CONFIG = TunableConfig()

DEFAULT_STATE = {
    "stage": "egg",
    "evolution_tier": 0,
    "incubation_progress": 0,
    "hunger": 50,
    "happiness": 50,
    "weight": 50,
    "mood": "neutral",
    "born_at": None,
    "died_at": None,
    "comment_history": [],
    "last_event_at": None,
    "last_decay_at": None,
    "variant": "blob",
    "personality": "",
    "egg_captions": [],
    "variant_determined": False,
    "last_comment_qwen": False,
    "poop": 0,
    "hygiene": 100,
    "sickness": 0,
    "boredom": 0,
    "age_minutes": 0,
    "lifespan_minutes": 10080,
    "stats": {
        "total_images_eaten": 0,
        "images_this_life": 0,
        "total_love_received": 0,
        "generations_lived": 1,
    },
}

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

class GotchiState:
    def __init__(self, data=None):
        d = DEFAULT_STATE.copy()
        d["stats"] = DEFAULT_STATE["stats"].copy()
        d["comment_history"] = list(DEFAULT_STATE["comment_history"])
        d["egg_captions"] = list(DEFAULT_STATE["egg_captions"])
        if data:
            d.update(data)
            if "stats" in data:
                merged = DEFAULT_STATE["stats"].copy()
                merged.update(data["stats"])
                d["stats"] = merged
            if "comment_history" in data:
                d["comment_history"] = list(data["comment_history"])
            if "egg_captions" in data:
                d["egg_captions"] = list(data["egg_captions"])
        for k, v in d.items():
            setattr(self, k, v)
        self._cfg = CONFIG

    def to_dict(self):
        keys = list(DEFAULT_STATE.keys())
        return {k: getattr(self, k) for k in keys}

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def derive_mood(self):
        if self.stage == "ghost":
            return "dead"
        if self.stage == "egg":
            return "incubating"
        if self.sickness >= 60:
            return "sick"
        h = self.hunger
        j = self.happiness
        if h >= 80:
            return "miserable"
        if h >= 60:
            return "grumpy"
        if j >= 80 and h < 30:
            return "ecstatic"
        if j >= 60:
            return "happy"
        if self.boredom >= 70:
            return "grumpy"
        return "neutral"

    def apply_feed(self):
        self.last_event_at = _now_iso()
        self.stats["total_images_eaten"] += 1
        self.stats["images_this_life"] += 1
        if self.stage == "egg":
            self.incubation_progress += 1
            if self.incubation_progress >= self._cfg.HATCH_THRESHOLD:
                self._hatch()
            self.mood = self.derive_mood()
            return
        if self.stage == "ghost":
            self.mood = self.derive_mood()
            return
        self.hunger = max(0, self.hunger - self._cfg.DELTA_FEED)
        self.happiness = min(100, self.happiness + self._cfg.DELTA_FEED * 0.3)
        self.weight = min(100, self.weight + self._cfg.DELTA_FEED * 0.2)
        if self.weight > 90:
            self.sickness = min(100, self.sickness + 1)
        import random as _r
        if _r.random() < self._cfg.DELTA_POOP_CHANCE:
            self.poop = min(self._cfg.POOP_MAX, self.poop + 1)
        self._check_growup()
        self._check_evolution()
        self._check_death()
        self.mood = self.derive_mood()

    def apply_love(self):
        self.last_event_at = _now_iso()
        self.stats["total_love_received"] += 1
        if self.stage in ("egg", "ghost"):
            self.mood = self.derive_mood()
            return
        self.happiness = min(100, self.happiness + self._cfg.DELTA_LOVE)
        self.boredom = max(0, self.boredom - 5)
        self.mood = self.derive_mood()

    def apply_play(self):
        self.last_event_at = _now_iso()
        if self.stage in ("egg", "ghost"):
            self.mood = self.derive_mood()
            return
        self.happiness = min(100, self.happiness + self._cfg.PLAY_HAPPINESS_BOOST)
        self.boredom = max(0, self.boredom - 30)
        self.hunger = min(100, self.hunger + 3)
        self.mood = self.derive_mood()

    def apply_clean(self):
        self.last_event_at = _now_iso()
        if self.stage in ("egg", "ghost"):
            self.mood = self.derive_mood()
            return
        self.poop = 0
        self.hygiene = min(100, self.hygiene + self._cfg.CLEAN_HYGIENE_BOOST)
        self.happiness = min(100, self.happiness + 5)
        self.mood = self.derive_mood()

    def apply_medicine(self):
        self.last_event_at = _now_iso()
        if self.stage in ("egg", "ghost"):
            self.mood = self.derive_mood()
            return
        self.sickness = self._cfg.MEDICINE_SICKNESS_RESET
        self.happiness = max(0, self.happiness - 10)
        self.mood = self.derive_mood()

    def apply_tick(self, elapsed_minutes):
        self.last_event_at = _now_iso()
        if self.stage in ("egg", "ghost"):
            self.last_decay_at = _now_iso()
            return
        em = elapsed_minutes
        self.hunger = min(100, self.hunger + self._cfg.DELTA_HUNGER_PER_MIN * em)
        self.happiness = max(0, self.happiness - self._cfg.DELTA_HAPPINESS_DECAY_PER_MIN * em)
        self.boredom = min(100, self.boredom + self._cfg.DELTA_BOREDOM_PER_MIN * em)
        self.hygiene = max(0, self.hygiene - self._cfg.DELTA_HYGIENE_DECAY_PER_MIN * em)
        self.age_minutes += em
        if self.poop > 0:
            self.sickness = min(100, self.sickness + self._cfg.DELTA_SICKNESS_FROM_POOP * em * (self.poop / self._cfg.POOP_MAX))
        if self.boredom > 50:
            self.sickness = min(100, self.sickness + self._cfg.DELTA_SICKNESS_FROM_BOREDOM * em)
        if self.hunger > 80:
            self.sickness = min(100, self.sickness + self._cfg.DELTA_SICKNESS_FROM_HUNGER * em)
        if self.age_minutes > self.lifespan_minutes * 0.7:
            age_factor = (self.age_minutes - self.lifespan_minutes * 0.7) / (self.lifespan_minutes * 0.3)
            self.sickness = min(100, self.sickness + self._cfg.DELTA_SICKNESS_FROM_AGE * em * age_factor)
        self.last_decay_at = _now_iso()
        self._check_death()
        self.mood = self.derive_mood()

    def _hatch(self):
        import random as _r
        self.stage = "hatchling"
        self.born_at = _now_iso()
        self.hunger = 40
        self.happiness = 60
        self.weight = 40
        self.incubation_progress = 0
        self.poop = 0
        self.hygiene = 100
        self.sickness = 0
        self.boredom = 0
        self.age_minutes = 0
        self.lifespan_minutes = _r.randint(self._cfg.AGE_MIN_MINUTES, self._cfg.AGE_MAX_MINUTES)

    def _check_growup(self):
        if self.stage == "hatchling" and self.stats["images_this_life"] >= self._cfg.HATCH_THRESHOLD + self._cfg.N_FEEDS_GROWUP:
            self.stage = "adult"

    def _check_evolution(self):
        tier = self.stats["total_images_eaten"] // self._cfg.EVOLUTION_THRESHOLD
        if tier > self.evolution_tier:
            self.evolution_tier = tier
            if self.stage in ("hatchling", "adult"):
                self.stage = "evolved"

    def _check_death(self):
        if self.stage == "ghost":
            return
        if self.sickness >= self._cfg.SICKNESS_DEATH_THRESHOLD:
            self.stage = "ghost"
            self.died_at = _now_iso()
            self.mood = "dead"
            return
        if self.hunger >= 100:
            self.stage = "ghost"
            self.died_at = _now_iso()
            self.mood = "dead"
            return
        if self.age_minutes >= self.lifespan_minutes:
            self.stage = "ghost"
            self.died_at = _now_iso()
            self.mood = "dead"

    def check_reincarnation(self, ghost_minutes, ghost_events):
        if self.stage != "ghost":
            return False
        if ghost_minutes >= self._cfg.T_GHOST_MIN or ghost_events >= self._cfg.GHOST_EVENT_THRESHOLD:
            self.stage = "egg"
            self.incubation_progress = 0
            self.hunger = 50
            self.happiness = 50
            self.weight = 50
            self.died_at = None
            self.born_at = None
            self.variant = "blob"
            self.personality = ""
            self.egg_captions = []
            self.variant_determined = False
            self.poop = 0
            self.hygiene = 100
            self.sickness = 0
            self.boredom = 0
            self.age_minutes = 0
            self.lifespan_minutes = 10080
            self.stats["images_this_life"] = 0
            self.stats["generations_lived"] += 1
            self.mood = self.derive_mood()
            return True
        return False

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.to_dict()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return cls()
