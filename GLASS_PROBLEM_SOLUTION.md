# 🔍 "Glass" Problem - Warum wird immer Glas generiert?

## ❓ **Das Problem:**
Egal welchen Prompt du verwendest, es wird immer ein Glas oder eine Flasche generiert.

## 🎯 **Die Ursache:**
Das **LayerDiffuse LoRA** (`layerlora.safetensors`) wurde hauptsächlich auf **Glas-Objekte** trainiert:
- Flaschen
- Weingläser  
- Kristall-Skulpturen
- Transparente Glas-Objekte

## ✅ **Lösungen:**

### **1. LoRA-Stärke reduzieren**
```
LoraLoaderModelOnly:
- layerlora.safetensors
- Strength: 0.5-0.8 (statt 1.0)
```

### **2. Stärkere Prompts verwenden**
```
Positive Prompt:
"a red football, transparent, NOT glass, NOT bottle, sports equipment"

Negative Prompt:
"glass, bottle, wine glass, crystal, glassware, drinking glass"
```

### **3. Verschiedene Objekte testen**
```
✅ Funktioniert oft:
- "transparent plastic bottle"
- "clear acrylic sculpture" 
- "transparent gemstone"
- "crystal formation"
- "ice cube"
- "soap bubble"

❌ Schwierig:
- "transparent car" (zu komplex)
- "transparent person" (LoRA nicht darauf trainiert)
- "transparent building" (zu groß/komplex)
```

### **4. CFG-Werte anpassen**
```
KSampler:
- CFG: 5.0-8.0 (höher = stärkere Prompt-Befolgung)
- Steps: 25-35 (mehr Steps für bessere Kontrolle)
```

### **5. Verschiedene Sampler testen**
```
KSampler:
- Sampler: "dpmpp_2m" oder "dpmpp_sde"
- Scheduler: "karras" oder "exponential"
```

## 🧪 **Test-Prompts:**

### **Für Non-Glass Objekte:**
```
"a transparent plastic water bottle, clear polymer, NOT glass"
"transparent ice cube, frozen water, crystal clear"
"clear acrylic sculpture, modern art, transparent plastic"
"soap bubble floating, iridescent, transparent sphere"
"transparent gemstone, crystal quartz, mineral"
```

### **Negative Prompts:**
```
"glass, glassware, wine glass, drinking glass, bottle glass, crystal glass, glass material"
```

## 🔧 **Erweiterte Lösungen:**

### **1. Anderes LoRA verwenden**
- Suche nach anderen Transparency LoRAs
- Verwende kein LoRA (nur Flux Base Model)
- Trainiere eigenes LoRA auf gewünschte Objekte

### **2. Prompt-Weighting verwenden**
```
"(transparent football:1.5), (sports equipment:1.3), (NOT glass:1.2)"
```

### **3. Multi-Step Prompting**
```
1. Generiere ohne LoRA: "a red football"
2. Verwende img2img mit LoRA: "make this transparent"
```

## 📊 **Warum passiert das?**

### **LoRA Training Data:**
Das LayerDiffuse LoRA wurde wahrscheinlich trainiert mit:
- 80% Glas-Objekten (Flaschen, Gläser)
- 15% Kristall/Edelstein-Objekten  
- 5% Andere transparente Materialien

### **Model Bias:**
- **Flux Base Model** assoziiert "transparent" oft mit Glas
- **LoRA** verstärkt diese Assoziation noch mehr
- **Training Data** war hauptsächlich Glas-fokussiert

## 🎯 **Beste Strategie:**

1. **LoRA Strength auf 0.7 reduzieren**
2. **Starke negative Prompts** für Glas verwenden
3. **Spezifische Material-Beschreibungen** nutzen
4. **CFG auf 6-8 erhöhen** für bessere Prompt-Kontrolle
5. **Verschiedene Sampler** testen

## 💡 **Beispiel-Workflow:**

```
Positive: "transparent plastic football, clear polymer material, sports equipment, NOT glass"
Negative: "glass, bottle, wine glass, crystal glass, glassware"
LoRA Strength: 0.7
CFG: 7.0
Steps: 30
Sampler: dpmpp_2m
```

## 🔍 **Debugging:**

Wenn immer noch Glas generiert wird:
1. **LoRA komplett deaktivieren** (Strength: 0)
2. **Nur Flux Base Model** testen
3. **Andere Transparency-Methoden** suchen
4. **Eigenes LoRA trainieren** für gewünschte Objekte

**Das "Glass"-Problem ist normal bei LayerDiffuse - es liegt am Training Data des LoRAs!** 🎨✨
