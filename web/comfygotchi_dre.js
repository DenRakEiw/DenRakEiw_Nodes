import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TICK_INTERVAL_MS = 60000;
const POLL_INTERVAL_MS = 2000;

const NODE_W = 220;
const NODE_H = 420;
const DEVICE_OFFSET_Y = 120;

const W = 220;
const H = 290;

const SHELL_COLOR = "#f0e6d3";
const SHELL_DARK = "#d4c4a8";
const SHELL_BUTTON = "#c9b896";
const SHELL_BUTTON_HOVER = "#a89878";
const SCREEN_BG = "#9bbc0f";
const SCREEN_BG_DARK = "#8bac0f";
const PD = "#0f380f";
const PM = "#306230";
const PL = "#8bac0f";
const PW = "#c4cfa1";
const PURPLE = "#7c5fb8";
const RED = "#b33a3a";
const BROWN = "#6b4226";
const SICK_GREEN = "#5a8a3a";

let lastState = null;
let lastTickSent = 0;
let animFrame = 0;
let hoverButton = -1;
let lastComment = "";
let commentTimer = 0;
let lastCommentHash = "";

const BUTTONS = [
  { x: 45, y: 270, r: 12, label: "PLAY", emoji: "🎾", event: "play" },
  { x: 110, y: 270, r: 12, label: "CLEAN", emoji: "🧹", event: "clean" },
  { x: 175, y: 270, r: 12, label: "MEDS", emoji: "💊", event: "medicine" },
];

async function fetchState() {
  try {
    const r = await fetch("/comfygotchi_dre/state");
    lastState = await r.json();
    if (lastState && lastState.comment_history && lastState.comment_history.length > 0) {
      const latest = lastState.comment_history[lastState.comment_history.length - 1];
      const hash = JSON.stringify(latest);
      if (hash !== lastCommentHash && latest) {
        lastCommentHash = hash;
        lastComment = typeof latest === "string" ? latest : (latest.comment || latest.text || "");
        commentTimer = lastState.last_comment_qwen ? 600 : 300;
      }
    }
    return lastState;
  } catch (e) {
    console.warn("[ComfyGotchi_DRE] fetchState failed", e);
    return null;
  }
}

async function sendTick() {
  const now = Date.now();
  if (now - lastTickSent < TICK_INTERVAL_MS - 1000) return;
  lastTickSent = now;
  try {
    await fetch("/comfygotchi_dre/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "tick", elapsed_minutes: 1 }),
    });
  } catch (e) {
    console.warn("[ComfyGotchi_DRE] tick failed", e);
  }
}

async function sendAction(eventType) {
  try {
    const r = await fetch("/comfygotchi_dre/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: eventType }),
    });
    lastState = await r.json();
  } catch (e) {
    console.warn(`[ComfyGotchi_DRE] ${eventType} failed`, e);
  }
}

function px(ctx, x, y, w, h, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
}

function drawEgg(ctx, cx, cy, crackProgress, bob) {
  const oy = Math.round(bob);
  const ex = cx;
  const ey = cy + oy;
  const shake = crackProgress > 0.8 ? Math.round(Math.sin(animFrame * 0.3) * 2) : 0;
  const fx = ex + shake;

  for (let dy = -18; dy <= 18; dy++) {
    for (let dx = -14; dx <= 14; dx++) {
      const dist = Math.sqrt((dx * dx) / (14 * 14) + ((dy + 4) * (dy + 4)) / (18 * 18));
      if (dist <= 1) {
        let c = PW;
        if (dist > 0.85) c = PL;
        if (dist > 0.95) c = PM;
        px(ctx, fx + dx, ey + dy, 1, 1, c);
      }
    }
  }
  // Decorative spots (no eyes/mouth — it's an egg, not a creature yet)
  px(ctx, fx - 7, ey - 8, 2, 1, PL);
  px(ctx, fx - 6, ey - 7, 1, 1, PL);
  px(ctx, fx + 5, ey - 4, 2, 2, PL);
  px(ctx, fx + 6, ey - 3, 1, 1, PL);
  px(ctx, fx - 8, ey + 6, 1, 2, PL);
  px(ctx, fx + 4, ey + 8, 3, 1, PL);
  px(ctx, fx + 5, ey + 9, 1, 1, PL);
  if (crackProgress > 0.3) {
    px(ctx, fx - 6, ey - 10, 1, 3, PD);
    px(ctx, fx - 5, ey - 8, 2, 1, PD);
  }
  if (crackProgress > 0.5) {
    px(ctx, fx + 2, ey - 12, 1, 4, PD);
    px(ctx, fx + 3, ey - 9, 2, 1, PD);
  }
  if (crackProgress > 0.7) {
    px(ctx, fx - 8, ey - 14, 16, 1, PD);
    px(ctx, fx - 4, ey - 15, 8, 1, PD);
    px(ctx, fx + 5, ey - 8, 1, 3, PD);
  }
  if (crackProgress >= 0.9) {
    const flash = Math.sin(animFrame * 0.5) > 0;
    if (flash) {
      for (let dy = -20; dy <= 20; dy++) {
        for (let dx = -16; dx <= 16; dx++) {
          if (Math.abs(dx) + Math.abs(dy) < 16) {
            px(ctx, fx + dx, ey + dy, 1, 1, PW);
          }
        }
      }
    }
  }
}

function drawBodyBase(ctx, cx, cy, rx, ry, fillColor, edgeColor, tier) {
  for (let dy = -ry; dy <= ry; dy++) {
    for (let dx = -rx; dx <= rx; dx++) {
      const dist = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
      if (dist <= 1) {
        let c = fillColor;
        if (dist > 0.8) c = edgeColor;
        if (tier > 0 && dist < 0.3) c = PURPLE;
        px(ctx, cx + dx, cy + dy, 1, 1, c);
      }
    }
  }
}

function drawEyes(ctx, cx, cy, mood, sep) {
  const eyeY = cy;
  if (mood === "sick") {
    px(ctx, cx - sep - 1, eyeY, 3, 1, PD);
    px(ctx, cx - sep, eyeY + 1, 1, 1, PD);
    px(ctx, cx + sep - 1, eyeY, 3, 1, PD);
    px(ctx, cx + sep, eyeY + 1, 1, 1, PD);
  } else if (mood === "grumpy" || mood === "miserable") {
    px(ctx, cx - sep - 1, eyeY - 1, 3, 1, PD);
    px(ctx, cx + sep - 1, eyeY - 1, 3, 1, PD);
  } else {
    px(ctx, cx - sep, eyeY, 2, 2, PD);
    px(ctx, cx + sep - 1, eyeY, 2, 2, PD);
  }
}

function drawMouth(ctx, cx, cy, mood, width) {
  if (mood === "sick") {
    px(ctx, cx - width, cy + 5, width * 2, 1, PD);
    px(ctx, cx - 1, cy + 4, 2, 1, PD);
    return;
  }
  if (mood === "happy" || mood === "ecstatic") {
    px(ctx, cx - width, cy + 4, width * 2, 1, PD);
    px(ctx, cx - width - 1, cy + 3, 1, 1, PD);
    px(ctx, cx + width, cy + 3, 1, 1, PD);
  } else if (mood === "miserable") {
    px(ctx, cx - width, cy + 6, width * 2, 1, PD);
    px(ctx, cx - width - 1, cy + 7, 1, 1, PD);
    px(ctx, cx + width, cy + 7, 1, 1, PD);
  } else {
    px(ctx, cx - width, cy + 4, width * 2, 1, PD);
  }
}

function drawSickBubble(ctx, cx, cy) {
  const bx = cx + 14;
  const by = cy - 14 + Math.sin(animFrame * 0.1) * 2;
  px(ctx, bx, by, 2, 2, SICK_GREEN);
  px(ctx, bx + 3, by + 1, 1, 1, SICK_GREEN);
  px(ctx, bx - 2, by + 3, 1, 1, SICK_GREEN);
}

function drawPoop(ctx, cx, baseY, count) {
  for (let i = 0; i < count; i++) {
    const px2 = cx - 30 + i * 14;
    const py = baseY;
    px(ctx, px2, py, 4, 2, BROWN);
    px(ctx, px2 - 1, py + 2, 6, 2, BROWN);
    px(ctx, px2, py + 4, 4, 1, BROWN);
    px(ctx, px2 + 1, py - 1, 2, 1, BROWN);
  }
}

function drawBlob(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 16 + Math.round(weight * 0.1);
  const ry = 14;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, PW, PL, tier);
  // antenna
  px(ctx, cx, cy + oy - ry - 5, 1, 5, PD);
  px(ctx, cx - 1, cy + oy - ry - 6, 3, 1, PD);
  drawEyes(ctx, cx, cy + oy - 2, mood, 5);
  drawMouth(ctx, cx, cy + oy, mood, 4);
}

function drawCat(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 15 + Math.round(weight * 0.08);
  const ry = 13;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, PW, PL, tier);
  // BIG pointy triangle ears
  px(ctx, cx - rx + 1, cy + oy - ry + 2, 2, 8, PW);
  px(ctx, cx - rx - 1, cy + oy - ry, 4, 6, PW);
  px(ctx, cx - rx - 2, cy + oy - ry - 2, 5, 4, PW);
  px(ctx, cx - rx - 1, cy + oy - ry - 4, 3, 2, PW);
  px(ctx, cx + rx - 3, cy + oy - ry + 2, 2, 8, PW);
  px(ctx, cx + rx - 1, cy + oy - ry, 4, 6, PW);
  px(ctx, cx + rx - 2, cy + oy - ry - 2, 5, 4, PW);
  px(ctx, cx + rx - 1, cy + oy - ry - 4, 3, 2, PW);
  // inner ear pink
  px(ctx, cx - rx, cy + oy - ry + 1, 2, 3, "#e8a0a0");
  px(ctx, cx + rx - 1, cy + oy - ry + 1, 2, 3, "#e8a0a0");
  // whiskers
  px(ctx, cx - rx - 3, cy + oy + 2, 5, 1, PD);
  px(ctx, cx + rx - 2, cy + oy + 2, 5, 1, PD);
  px(ctx, cx - rx - 2, cy + oy + 4, 4, 1, PD);
  px(ctx, cx + rx - 1, cy + oy + 4, 4, 1, PD);
  drawEyes(ctx, cx, cy + oy - 2, mood, 5);
  // pink nose
  px(ctx, cx - 1, cy + oy + 2, 3, 1, "#e8a0a0");
  px(ctx, cx, cy + oy + 3, 1, 1, "#e8a0a0");
  drawMouth(ctx, cx, cy + oy + 2, mood, 3);
}

function drawDog(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 16 + Math.round(weight * 0.1);
  const ry = 13;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, PW, PL, tier);
  // BIG floppy ears hanging down on sides
  px(ctx, cx - rx - 3, cy + oy - ry + 4, 4, 12, "#c9a878");
  px(ctx, cx - rx - 4, cy + oy - ry + 6, 5, 10, "#c9a878");
  px(ctx, cx - rx - 3, cy + oy - ry + 8, 4, 8, "#b89868");
  px(ctx, cx + rx, cy + oy - ry + 4, 4, 12, "#c9a878");
  px(ctx, cx + rx + 1, cy + oy - ry + 6, 5, 10, "#c9a878");
  px(ctx, cx + rx, cy + oy - ry + 8, 4, 8, "#b89868");
  // brown snout patch
  px(ctx, cx - 4, cy + oy + 2, 9, 5, "#c9a878");
  // black nose
  px(ctx, cx - 1, cy + oy + 3, 3, 2, PD);
  // tongue out if happy
  if (mood === "happy" || mood === "ecstatic") {
    px(ctx, cx - 1, cy + oy + 6, 3, 3, "#e85858");
  }
  drawEyes(ctx, cx, cy + oy - 3, mood, 5);
  drawMouth(ctx, cx, cy + oy + 5, mood, 3);
}

function drawMonster(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 16 + Math.round(weight * 0.1);
  const ry = 14;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, "#9b7fc4", "#7c5fb8", tier);
  // BIG horns
  px(ctx, cx - rx + 2, cy + oy - ry - 1, 1, 6, PD);
  px(ctx, cx - rx + 1, cy + oy - ry - 3, 3, 4, PD);
  px(ctx, cx - rx, cy + oy - ry - 5, 2, 2, PD);
  px(ctx, cx + rx - 3, cy + oy - ry - 1, 1, 6, PD);
  px(ctx, cx + rx - 3, cy + oy - ry - 3, 3, 4, PD);
  px(ctx, cx + rx - 2, cy + oy - ry - 5, 2, 2, PD);
  // angry eyebrows
  px(ctx, cx - 7, cy + oy - 5, 5, 1, PD);
  px(ctx, cx - 8, cy + oy - 4, 3, 1, PD);
  px(ctx, cx + 3, cy + oy - 5, 5, 1, PD);
  px(ctx, cx + 6, cy + oy - 4, 3, 1, PD);
  drawEyes(ctx, cx, cy + oy - 2, mood, 5);
  // sharp teeth row
  px(ctx, cx - 6, cy + oy + 4, 2, 1, PW);
  px(ctx, cx - 5, cy + oy + 5, 1, 1, PW);
  px(ctx, cx - 3, cy + oy + 4, 2, 2, PW);
  px(ctx, cx, cy + oy + 4, 2, 1, PW);
  px(ctx, cx + 1, cy + oy + 5, 1, 1, PW);
  px(ctx, cx + 3, cy + oy + 4, 2, 2, PW);
  px(ctx, cx + 5, cy + oy + 5, 1, 1, PW);
}

function drawDragon(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 16 + Math.round(weight * 0.1);
  const ry = 12;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, "#7da870", "#5a8a3a", tier);
  // BIG wings
  px(ctx, cx - rx - 6, cy + oy - 5, 8, 3, "#5a8a3a");
  px(ctx, cx - rx - 7, cy + oy - 7, 6, 3, "#5a8a3a");
  px(ctx, cx - rx - 5, cy + oy - 9, 4, 3, "#5a8a3a");
  px(ctx, cx + rx + 1, cy + oy - 5, 8, 3, "#5a8a3a");
  px(ctx, cx + rx + 2, cy + oy - 7, 6, 3, "#5a8a3a");
  px(ctx, cx + rx + 2, cy + oy - 9, 4, 3, "#5a8a3a");
  // horns
  px(ctx, cx - rx + 3, cy + oy - ry - 1, 1, 5, PD);
  px(ctx, cx - rx + 2, cy + oy - ry - 3, 3, 3, PD);
  px(ctx, cx + rx - 4, cy + oy - ry - 1, 1, 5, PD);
  px(ctx, cx + rx - 4, cy + oy - ry - 3, 3, 3, PD);
  // tail with spikes
  px(ctx, cx + rx + 3, cy + oy + ry - 2, 6, 2, "#5a8a3a");
  for (let i = 0; i < 4; i++) {
    px(ctx, cx + rx + 4 + i * 2, cy + oy + ry - 4, 1, 2, PD);
  }
  // scale texture
  px(ctx, cx - 5, cy + oy - 2, 2, 1, "#4a7a2a");
  px(ctx, cx + 3, cy + oy - 2, 2, 1, "#4a7a2a");
  px(ctx, cx - 2, cy + oy + 3, 2, 1, "#4a7a2a");
  drawEyes(ctx, cx, cy + oy - 2, mood, 5);
  // flame breath if ecstatic
  if (mood === "ecstatic") {
    px(ctx, cx - 3, cy + oy + 5, 7, 2, "#e85820");
    px(ctx, cx - 2, cy + oy + 7, 5, 2, "#e8a020");
  }
  drawMouth(ctx, cx, cy + oy + 2, mood, 4);
}

function drawRobot(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rw = 18 + Math.round(weight * 0.1);
  const rh = 15;
  const ex = cx;
  const ey = cy + oy;
  // rectangular body
  for (let dy = -rh; dy <= rh; dy++) {
    for (let dx = -rw; dx <= rw; dx++) {
      if (Math.abs(dx) <= rw && Math.abs(dy) <= rh) {
        let c = "#b8b8c8";
        if (Math.abs(dx) > rw - 2 || Math.abs(dy) > rh - 2) c = "#9090a0";
        if (tier > 0 && Math.abs(dx) < 5 && Math.abs(dy) < 5) c = PURPLE;
        px(ctx, ex + dx, ey + dy, 1, 1, c);
      }
    }
  }
  // BIG antenna with blinking light
  px(ctx, ex, ey - rh - 6, 1, 6, PD);
  px(ctx, ex - 2, ey - rh - 7, 5, 1, PD);
  const blink = Math.sin(animFrame * 0.15) > 0;
  px(ctx, ex, ey - rh - 8, 1, 1, blink ? RED : "#602020");
  // LED eyes (rectangular)
  if (mood === "sick" || mood === "grumpy" || mood === "miserable") {
    px(ctx, ex - 7, ey - 3, 4, 2, RED);
    px(ctx, ex + 3, ey - 3, 4, 2, RED);
  } else {
    px(ctx, ex - 7, ey - 3, 4, 3, "#20e020");
    px(ctx, ex + 3, ey - 3, 4, 3, "#20e020");
  }
  // chest panel
  px(ctx, ex - 4, ey + 3, 9, 6, "#404050");
  px(ctx, ex - 3, ey + 4, 7, 1, "#20e020");
  px(ctx, ex - 3, ey + 6, 3, 1, "#e0e020");
  px(ctx, ex + 1, ey + 6, 3, 1, RED);
  // side bolts
  px(ctx, ex - rw - 1, ey - 2, 2, 4, "#606070");
  px(ctx, ex + rw, ey - 2, 2, 4, "#606070");
}

function drawPhantom(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob * 0.5);
  const rx = 15 + Math.round(weight * 0.08);
  const ry = 14;
  const ex = cx;
  const ey = cy + oy;
  ctx.globalAlpha = 0.7;
  // ghostly rounded top
  for (let dy = -ry; dy <= 0; dy++) {
    for (let dx = -rx; dx <= rx; dx++) {
      const dist = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
      if (dist <= 1) {
        let c = PW;
        if (dist > 0.8) c = PL;
        if (tier > 0 && dist < 0.3) c = PURPLE;
        px(ctx, ex + dx, ey + dy, 1, 1, c);
      }
    }
  }
  // wavy bottom
  const waveY = ey + 1;
  for (let dx = -rx; dx <= rx; dx++) {
    const wave = Math.round(Math.sin((dx + animFrame * 0.1) * 0.5) * 3);
    px(ctx, ex + dx, waveY + wave, 1, 8, PW);
  }
  px(ctx, ex - rx, waveY + 5, 4, 4, PW);
  px(ctx, ex - 4, waveY + 5, 4, 5, PW);
  px(ctx, ex + 1, waveY + 5, 4, 4, PW);
  ctx.globalAlpha = 1.0;
  drawEyes(ctx, ex, ey - 2, mood, 5);
  drawMouth(ctx, ex, ey, mood, 4);
}

function drawAlien(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 13 + Math.round(weight * 0.08);
  const ry = 15;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, "#7cc070", "#4a8a3a", tier);
  // BIG antenna with glowing tip
  px(ctx, cx, cy + oy - ry - 5, 1, 5, PM);
  px(ctx, cx - 2, cy + oy - ry - 6, 5, 1, PM);
  const glow = Math.sin(animFrame * 0.2) > 0;
  px(ctx, cx, cy + oy - ry - 7, 1, 1, glow ? "#20e020" : "#4a8a3a");
  // BIG almond eyes
  px(ctx, cx - 6, cy + oy - 3, 4, 3, PD);
  px(ctx, cx - 5, cy + oy - 2, 2, 1, PW);
  px(ctx, cx + 3, cy + oy - 3, 4, 3, PD);
  px(ctx, cx + 4, cy + oy - 2, 2, 1, PW);
  // tiny mouth
  px(ctx, cx - 1, cy + oy + 4, 3, 1, PD);
  // arms
  px(ctx, cx - rx - 2, cy + oy + 2, 3, 1, "#4a8a3a");
  px(ctx, cx + rx, cy + oy + 2, 3, 1, "#4a8a3a");
}

function drawBunny(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 14 + Math.round(weight * 0.08);
  const ry = 12;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, PW, PL, tier);
  // VERY long ears
  px(ctx, cx - 7, cy + oy - ry - 12, 4, 14, PW);
  px(ctx, cx - 6, cy + oy - ry - 10, 2, 10, "#e8a0a0");
  px(ctx, cx + 4, cy + oy - ry - 12, 4, 14, PW);
  px(ctx, cx + 5, cy + oy - ry - 10, 2, 10, "#e8a0a0");
  // ear twitch
  const twitch = Math.sin(animFrame * 0.08) > 0.5 ? 1 : 0;
  px(ctx, cx + 4 + twitch, cy + oy - ry - 12, 4, 2, PW);
  drawEyes(ctx, cx, cy + oy - 2, mood, 4);
  // pink nose
  px(ctx, cx - 1, cy + oy + 2, 3, 1, "#e8a0a0");
  drawMouth(ctx, cx, cy + oy + 4, mood, 2);
  // buck teeth
  px(ctx, cx - 1, cy + oy + 5, 1, 2, PW);
  px(ctx, cx, cy + oy + 5, 1, 2, PW);
  // feet
  px(ctx, cx - 9, cy + oy + ry - 1, 4, 2, PW);
  px(ctx, cx + 6, cy + oy + ry - 1, 4, 2, PW);
}

function drawPenguin(ctx, cx, cy, mood, bob, weight, tier) {
  const oy = Math.round(bob);
  const rx = 15 + Math.round(weight * 0.1);
  const ry = 15;
  drawBodyBase(ctx, cx, cy + oy, rx, ry, PD, PD, 0);
  // white belly
  for (let dy = -10; dy <= 8; dy++) {
    for (let dx = -9; dx <= 9; dx++) {
      if (Math.abs(dx) + Math.abs(dy) < 13) {
        px(ctx, cx + dx, cy + oy + dy, 1, 1, PW);
      }
    }
  }
  if (tier > 0) {
    px(ctx, cx, cy + oy - 2, 4, 4, PURPLE);
  }
  drawEyes(ctx, cx, cy + oy - 5, mood, 4);
  // BIG orange beak
  px(ctx, cx - 3, cy + oy - 1, 7, 3, "#e8a020");
  px(ctx, cx - 2, cy + oy + 1, 5, 1, "#c88010");
  // orange feet
  px(ctx, cx - 7, cy + oy + ry, 4, 2, "#e8a020");
  px(ctx, cx + 4, cy + oy + ry, 4, 2, "#e8a020");
  px(ctx, cx - 6, cy + oy + ry + 1, 2, 1, "#e8a020");
  px(ctx, cx + 5, cy + oy + ry + 1, 2, 1, "#e8a020");
  // flippers
  px(ctx, cx - rx - 1, cy + oy + 2, 3, 6, PD);
  px(ctx, cx + rx - 2, cy + oy + 2, 3, 6, PD);
}

function drawGhost(ctx, cx, cy, bob) {
  const oy = Math.round(bob * 0.5);
  const ex = cx;
  const ey = cy + oy;
  for (let dy = -12; dy <= 10; dy++) {
    for (let dx = -10; dx <= 10; dx++) {
      if (dx * dx + dy * dy <= 100) {
        px(ctx, ex + dx, ey + dy, 1, 1, PM);
      }
    }
  }
  px(ctx, ex - 10, ey + 10, 4, 2, PM);
  px(ctx, ex - 4, ey + 10, 4, 3, PM);
  px(ctx, ex + 2, ey + 10, 4, 2, PM);
  px(ctx, ex - 5, ey - 3, 2, 2, PD);
  px(ctx, ex + 3, ey - 3, 2, 2, PD);
}

function drawHearts(ctx, x, y, count, max) {
  for (let i = 0; i < max; i++) {
    const filled = i < count;
    const hx = x + i * 7;
    const c = filled ? PD : PL;
    px(ctx, hx, y, 1, 1, c);
    px(ctx, hx + 2, y, 1, 1, c);
    px(ctx, hx, y + 1, 3, 1, c);
    px(ctx, hx + 1, y + 2, 1, 1, c);
  }
}

function drawBar(ctx, x, y, value, max, width, color, bgColor) {
  const filled = Math.round((value / max) * width);
  for (let i = 0; i < width; i++) {
    const c = i < filled ? color : bgColor;
    px(ctx, x + i, y, 1, 3, c);
  }
}

function drawText(ctx, text, x, y, color = PD) {
  ctx.fillStyle = color;
  ctx.font = "6px monospace";
  ctx.textBaseline = "top";
  ctx.fillText(text, x, y);
}

function drawSpeechBubble(ctx, x, y, text) {
  if (!text) return;
  const maxW = 180;
  ctx.font = "7px monospace";
  const lines = [];
  const words = text.split(" ");
  let line = "";
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (ctx.measureText(test).width > maxW) {
      if (line) lines.push(line);
      line = w;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  const lineH = 9;
  const bubbleH = lines.length * lineH + 6;
  const bubbleW = maxW + 8;
  const bx = x - bubbleW / 2;
  const by = y - bubbleH - 4;

  ctx.fillStyle = PW;
  ctx.fillRect(bx, by, bubbleW, bubbleH);
  ctx.strokeStyle = PD;
  ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, bubbleW, bubbleH);
  ctx.fillStyle = PW;
  ctx.beginPath();
  ctx.moveTo(x - 3, by + bubbleH);
  ctx.lineTo(x + 3, by + bubbleH);
  ctx.lineTo(x, by + bubbleH + 4);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = PD;
  ctx.beginPath();
  ctx.moveTo(x - 3, by + bubbleH);
  ctx.lineTo(x, by + bubbleH + 4);
  ctx.lineTo(x + 3, by + bubbleH);
  ctx.stroke();

  ctx.fillStyle = PD;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], bx + 4, by + 3 + i * lineH);
  }
}

function drawCreature(ctx, state) {
  ctx.fillStyle = SHELL_COLOR;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = SHELL_DARK;
  ctx.fillRect(6, 6, W - 12, H - 12);
  ctx.fillStyle = SHELL_COLOR;
  ctx.fillRect(8, 8, W - 16, H - 16);

  const sx = 14;
  const sy = 16;
  const sw = W - 28;
  const sh = 140;

  ctx.fillStyle = SCREEN_BG;
  ctx.fillRect(sx, sy, sw, sh);
  ctx.fillStyle = SCREEN_BG_DARK;
  for (let i = 0; i < sw; i += 2) {
    for (let j = 0; j < sh; j += 2) {
      if ((i + j) % 4 === 0) {
        ctx.fillRect(sx + i, sy + j, 1, 1, SCREEN_BG_DARK);
      }
    }
  }
  ctx.strokeStyle = SHELL_DARK;
  ctx.lineWidth = 2;
  ctx.strokeRect(sx - 1, sy - 1, sw + 2, sh + 2);

  const cx = sx + Math.round(sw / 2);
  const cy = sy + Math.round(sh / 2) - 10;
  const bob = Math.sin(animFrame * 0.05) * 2;
  const stage = state ? (state.stage || "egg") : "egg";
  const mood = state ? (state.mood || "neutral") : "neutral";
  const weight = state ? (state.weight || 50) : 50;
  const tier = state ? (state.evolution_tier || 0) : 0;
  const variant = state ? (state.variant || "blob") : "blob";
  const sickness = state ? (state.sickness || 0) : 0;
  const poop = state ? (state.poop || 0) : 0;

  if (stage === "egg") {
    const crack = state ? (state.incubation_progress || 0) / 10 : 0;
    drawEgg(ctx, cx, cy, crack, bob);
  } else if (stage === "ghost") {
    drawGhost(ctx, cx, cy, bob);
  } else {
    const drawers = {
      blob: drawBlob, cat: drawCat, dog: drawDog, monster: drawMonster,
      dragon: drawDragon, robot: drawRobot, phantom: drawPhantom,
      alien: drawAlien, bunny: drawBunny, penguin: drawPenguin
    };
    const drawer = drawers[variant] || drawBlob;
    drawer(ctx, cx, cy, mood, bob, weight, tier);
    if (mood === "ecstatic") {
      const hx = cx + 16 + Math.sin(animFrame * 0.1) * 2;
      const hy = cy - 10 + Math.cos(animFrame * 0.15) * 2;
      px(ctx, hx, hy, 1, 1, PD);
      px(ctx, hx + 1, hy - 1, 1, 1, PD);
      px(ctx, hx - 1, hy - 1, 1, 1, PD);
      px(ctx, hx, hy - 2, 1, 1, PD);
    }
    if (sickness > 30) {
      drawSickBubble(ctx, cx, cy);
    }
  }

  if (poop > 0 && stage !== "egg" && stage !== "ghost") {
    drawPoop(ctx, cx, sy + sh - 12, poop);
  }

  if (commentTimer > 0 && lastComment) {
    drawSpeechBubble(ctx, cx, sy + 8, lastComment);
    commentTimer--;
  }

  const barY = sy + sh + 4;
  drawText(ctx, "HUN", sx + 2, barY, PD);
  drawBar(ctx, sx + 22, barY, state ? (state.hunger || 0) : 0, 100, 24, RED, PL);
  drawText(ctx, "JOY", sx + 2, barY + 7, PD);
  const joyCount = Math.round((state ? (state.happiness || 0) : 0) / 25);
  drawHearts(ctx, sx + 22, barY + 7, joyCount, 4);

  const bar2X = sx + 60;
  drawText(ctx, "HYG", bar2X, barY, PD);
  drawBar(ctx, bar2X + 18, barY, state ? (state.hygiene || 100) : 100, 100, 24, PD, PL);
  drawText(ctx, "SICK", bar2X, barY + 7, PD);
  drawBar(ctx, bar2X + 18, barY + 7, sickness, 100, 24, RED, PL);

  const eaten = state ? (state.stats?.total_images_eaten || 0) : 0;
  drawText(ctx, `M:${eaten}`, sx + 2, barY + 16, PD);
  if (tier > 0) {
    drawText(ctx, `T${tier}`, sx + 40, barY + 16, PD);
  }
  const ageMin = state ? (state.age_minutes || 0) : 0;
  drawText(ctx, `AGE:${Math.round(ageMin)}m`, bar2X, barY + 16, PD);

  const nextEvo = (tier + 1) * 5000;
  const progress = eaten / nextEvo;
  const barW = sw - 4;
  const filled = Math.round(progress * barW);
  for (let i = 0; i < barW; i++) {
    const c = i < filled ? PD : PL;
    px(ctx, sx + 2 + i, barY + 24, 1, 1, c);
  }

  for (let i = 0; i < BUTTONS.length; i++) {
    const btn = BUTTONS[i];
    const isHover = hoverButton === i;
    ctx.fillStyle = isHover ? SHELL_BUTTON_HOVER : SHELL_BUTTON;
    ctx.beginPath();
    ctx.arc(btn.x, btn.y, btn.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = SHELL_DARK;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = "10px serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(btn.emoji, btn.x, btn.y - 1);
    ctx.fillStyle = PD;
    ctx.font = "5px monospace";
    ctx.fillText(btn.label, btn.x, btn.y + 7);
    ctx.textAlign = "left";
  }
}

app.registerExtension({
  name: "comfygotchi_dre",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "ComfyGotchiNode") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      this.setSize([NODE_W, NODE_H]);
      this.onDrawBackground = function (ctx) {
        if (this.flags.collapsed) return;
        ctx.save();
        ctx.translate(0, DEVICE_OFFSET_Y);
        drawCreature(ctx, lastState);
        ctx.restore();
        animFrame++;
        this.setDirtyCanvas(true, false);
      };

      this.onMouseDown = function (e, pos, node) {
        if (!pos) return;
        const localX = pos[0];
        const localY = pos[1] - DEVICE_OFFSET_Y;
        for (let i = 0; i < BUTTONS.length; i++) {
          const btn = BUTTONS[i];
          const dx = localX - btn.x;
          const dy = localY - btn.y;
          if (dx * dx + dy * dy <= (btn.r + 4) * (btn.r + 4)) {
            sendAction(btn.event);
            const actionMsgs = {
              play: "Yay! Let's play!",
              clean: "All clean now!",
              medicine: "Ugh... but I feel better.",
            };
            lastComment = actionMsgs[btn.event] || "";
            commentTimer = 180;
            return true;
          }
        }
      };

      this.onMouseMove = function (e, pos, node) {
        if (!pos) return;
        const localX = pos[0];
        const localY = pos[1] - DEVICE_OFFSET_Y;
        let newHover = -1;
        for (let i = 0; i < BUTTONS.length; i++) {
          const btn = BUTTONS[i];
          const dx = localX - btn.x;
          const dy = localY - btn.y;
          if (dx * dx + dy * dy <= (btn.r + 4) * (btn.r + 4)) {
            newHover = i;
            break;
          }
        }
        if (newHover !== hoverButton) {
          hoverButton = newHover;
        }
      };

      return r;
    };
  },
  async setup() {
    await fetchState();
    setInterval(fetchState, POLL_INTERVAL_MS);
    setInterval(sendTick, TICK_INTERVAL_MS);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        fetchState();
        sendTick();
      }
    });
  },
});
