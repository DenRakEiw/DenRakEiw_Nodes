import random

VARIANTS = ["blob", "cat", "dog", "monster", "dragon", "robot", "phantom", "alien", "bunny", "penguin"]

TONES = {
    "dark": "snarky",
    "moody": "snarky",
    "gothic": "snarky",
    "noir": "snarky",
    "bright": "cheerful",
    "colorful": "cheerful",
    "happy": "cheerful",
    "nature": "calm",
    "organic": "calm",
    "natural": "calm",
    "sci-fi": "robotic",
    "cyber": "robotic",
    "tech": "robotic",
    "cute": "gentle",
    "kawaii": "gentle",
    "soft": "gentle",
    "horror": "morbid",
    "scary": "morbid",
    "creepy": "morbid",
}

TONAL_TEMPLATES = {
    "snarky": {
        "ecstatic": ["Oh joy. A {c}. How... unexpectedly tolerable.", "Fine. {c}. I'll allow it.", "Not terrible. For a {c}."],
        "happy": ["A {c}. Meh. Acceptable.", "{c}. Sure, I guess.", "Fine. {c}."],
        "neutral": ["{c}. Whatever.", "Another {c}. How original.", "{c}. Moving on."],
        "grumpy": ["A {c}? I'm starving and you bring me THIS?", "{c}. Disappointing.", "Is that all? A {c}?"],
        "miserable": ["{c}... the darkness consumes...", "Even this {c} cannot fill the void...", "{c}... pointless..."],
    },
    "cheerful": {
        "ecstatic": ["OH WOW! A {c}! This is AMAZING!", "Yes yes YES! {c}! I love it!", "A {c}! Best day EVER!"],
        "happy": ["Ooh, a {c}! So nice!", "Yay! {c}! Thank you!", "I love seeing a {c}!"],
        "neutral": ["A {c}. How lovely!", "Nice {c}!", "Pretty {c}!"],
        "grumpy": ["Hmm, a {c}. Could be better but okay.", "I want more than a {c}...", "A {c}? I suppose."],
        "miserable": ["Even a {c} can't cheer me up...", "{c}... I'm too hungry...", "Please... more than a {c}..."],
    },
    "calm": {
        "ecstatic": ["Ah, a {c}. Nature provides.", "The {c} speaks to me.", "A {c}. Harmony."],
        "happy": ["A {c}. Grounded. Good.", "I accept this {c}.", "The {c} is pleasant."],
        "neutral": ["A {c}. It is what it is.", "{c}. Present.", "I observe a {c}."],
        "grumpy": ["The {c} lacks substance.", "I need more than a {c}.", "A {c}. Insufficient."],
        "miserable": ["The {c} withers...", "Hunger clouds the {c}...", "{c}... fading..."],
    },
    "robotic": {
        "ecstatic": ["{c} DETECTED. EFFICIENCY OPTIMAL.", "INPUT: {c}. STATUS: ACCEPTABLE.", "{c} ANALYZED. SATISFACTION: 87%."],
        "happy": ["{c}. PROCESSING.", "INPUT ACCEPTED: {c}.", "{c}. ACKNOWLEDGED."],
        "neutral": ["{c}. LOGGED.", "INPUT: {c}.", "RECORDING {c}."],
        "grumpy": ["{c}. FUEL INSUFFICIENT.", "WARNING: {c} INADEQUATE.", "ENERGY LOW. {c} NOT ENOUGH."],
        "miserable": ["SYSTEM FAILURE. {c} UNABLE TO SUSTAIN.", "CRITICAL: {c}. SHUTDOWN IMMINENT.", "{c}... POWER... FAILING..."],
    },
    "gentle": {
        "ecstatic": ["Yay, a {c}~ So soft and nice!", "Ooh, {c}! I love it lots!", "A {c}! My heart is full!"],
        "happy": ["A {c}~ How sweet!", "{c}! Thank you kindly!", "I like this {c}!"],
        "neutral": ["A {c}. Okay~", "{c}. That's fine.", "Mm, a {c}."],
        "grumpy": ["I need more than a {c}...", "A {c}? But I'm hungry...", "More {c} please??"],
        "miserable": ["The {c} is too small...", "I'm fading... {c}...", "Please... more {c}..."],
    },
    "morbid": {
        "ecstatic": ["Ah, a {c}. Delicious suffering.", "The {c} pleases the darkness.", "Yes. A {c}. Feed the void."],
        "happy": ["A {c}. Acceptable sacrifice.", "The {c} will do.", "Mmm. {c}."],
        "neutral": ["Another {c} for the pile.", "{c}. Death comes for all.", "A {c}. How... mortal."],
        "grumpy": ["A pathetic {c}. I hunger for souls.", "{c}. Insufficient suffering.", "This {c} bores me."],
        "miserable": ["The {c} cannot save me...", "{c}... into the grave...", "Death... {c}... same..."],
    },
}

GENERIC_TEMPLATES = {
    "snarky": ["Oh another one. Thrilling.", "Wow. Again.", "I'm not impressed.", "Feeding me. How kind. Not."],
    "cheerful": ["Yum yum!", "Another meal! Yay!", "I love eating!", "More please!"],
    "calm": ["Another offering. Thank you.", "I accept this.", "The cycle continues.", "Present."],
    "robotic": ["INPUT RECEIVED.", "PROCESSING MEAL.", "FUEL INCREMENT.", "ACKNOWLEDGED."],
    "gentle": ["Thank you~", "Mm, more!", "So nice of you!", "I'm happy~"],
    "morbid": ["Another soul consumed.", "The void grows.", "More. Always more.", "Feed the darkness."],
}

GHOST_LINES = ["...", "boo.", "*floats silently*", "i was once alive...", "the void calls"]
SICK_LINES = ["*cough*", "i don't feel so good...", "*sneezes*", "ugh... my stomach...", "*wobbles weakly*"]
EGG_LINES = [""]

def _get_tone(personality):
    if not personality:
        return "snarky"
    p = personality.lower()
    for keyword, tone in TONES.items():
        if keyword in p:
            return tone
    return "snarky"

def generate_comment(mood, stage, evolution_tier, caption, personality="", variant="blob"):
    if stage == "egg":
        return ""
    if stage == "ghost":
        return random.choice(GHOST_LINES)
    if mood == "sick":
        return random.choice(SICK_LINES)
    
    tone = _get_tone(personality)
    tonal = TONAL_TEMPLATES.get(tone, TONAL_TEMPLATES["snarky"])
    pool = tonal.get(mood, tonal["neutral"])
    
    c = caption.strip() if caption else ""
    if c:
        c = c[:60]
        try:
            return random.choice(pool).format(c=c)
        except (KeyError, IndexError):
            return random.choice(pool).replace("{c}", c)
    else:
        generic = GENERIC_TEMPLATES.get(tone, GENERIC_TEMPLATES["snarky"])
        return random.choice(generic)
