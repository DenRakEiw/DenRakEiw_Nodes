import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let apiNodeTypes = null;
let apiNodePatterns = null;

const API_MODULE_PREFIXES = [
  "comfy_api_nodes",
  "comfy_extras.nodes_partner",
];

async function loadApiNodeTypes() {
  try {
    const r = await fetch("/object_info");
    const defs = await r.json();
    const types = new Set();
    const patterns = [];
    for (const [name, def] of Object.entries(defs)) {
      if (def.api_node === true) {
        types.add(name);
      }
      if (def.python_module) {
        for (const prefix of API_MODULE_PREFIXES) {
          if (def.python_module.startsWith(prefix)) {
            types.add(name);
            break;
          }
        }
      }
    }
    apiNodeTypes = types;
    console.log(`[ComfyGotchi_DRE] Love detection: ${types.size} API node types loaded`);
  } catch (e) {
    console.warn("[ComfyGotchi_DRE] Failed to load object_info for love detection", e);
    apiNodeTypes = null;
  }
}

async function sendLove() {
  try {
    await fetch("/comfygotchi_dre/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "love" }),
    });
  } catch (e) {
    console.warn("[ComfyGotchi_DRE] love event failed", e);
  }
}

function getNodeById(nodeId) {
  if (!nodeId) return null;
  const graph = app.graph || app.canvas?.graph;
  if (!graph) return null;
  if (graph._nodes_by_id) return graph._nodes_by_id[nodeId];
  if (graph._nodes) {
    for (const n of graph._nodes) {
      if (String(n.id) === String(nodeId)) return n;
    }
  }
  return null;
}

function isApiNode(node) {
  if (!node) return false;
  if (apiNodeTypes && apiNodeTypes.has(node.type)) return true;
  if (node.type) {
    const t = node.type.toLowerCase();
    if (t.includes("gemini") || t.includes("openai") || t.includes("nano") ||
        t.includes("banana") || t.includes("kling") || t.includes("luma") ||
        t.includes("ideogram") || t.includes("recraft") || t.includes("runway") ||
        t.includes("sora") || t.includes("veo") || t.includes("bfl") ||
        t.includes("bytedance") || t.includes("minimax") || t.includes("vidu") ||
        t.includes("topaz") || t.includes("tripo") || t.includes("meshy") ||
        t.includes("rodin") || t.includes("pixverse") || t.includes("wavespeed") ||
        t.includes("heygen") || t.includes("sync") || t.includes("sonilo") ||
        t.includes("magnific") || t.includes("reve") || t.includes("krea") ||
        t.includes("elevenlabs") || t.includes("anthropic") || t.includes("grok") ||
        t.includes("openrouter") || t.includes("beeble") || t.includes("quiver") ||
        t.includes("wan_api") || t.includes("ltxv_api")) {
      return true;
    }
  }
  return false;
}

function checkNodeAndSendLove(nodeId) {
  const node = getNodeById(String(nodeId));
  if (!node) return;
  if (isApiNode(node)) {
    console.log(`[ComfyGotchi_DRE] Love! ${node.type} executed`);
    sendLove();
  }
}

app.registerExtension({
  name: "comfygotchi_dre_listener",
  async setup() {
    await loadApiNodeTypes();
    setInterval(() => {
      if (apiNodeTypes === null) loadApiNodeTypes();
    }, 30000);

    api.addEventListener("executed", (evt) => {
      const detail = evt.detail || {};
      const nodeId = detail.node || detail.display_node;
      if (nodeId) checkNodeAndSendLove(nodeId);
    });

    api.addEventListener("execution_start", (evt) => {
      const graph = app.graph || app.canvas?.graph;
      if (!graph) return;
      const nodes = graph._nodes || [];
      for (const node of nodes) {
        if (isApiNode(node)) {
          sendLove();
          break;
        }
      }
    });
  },
});
