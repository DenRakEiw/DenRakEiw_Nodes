# 🔧 Flux LayerDiffuse - Tensor Dimension Error Fix

## ❌ **Problem:**
```
mat1 and mat2 shapes cannot be multiplied (77x2048 and 4096x3072)
```

## ✅ **Lösung:**

### **Methode 1: Conditioning Fix Node (Empfohlen)**

```
CLIP Text Encode → 🔧 Flux LayerDiffuse Conditioning Fix → KSampler
```

**Workflow:**
1. **CLIP Text Encode** → Dein Prompt
2. **🔧 Flux LayerDiffuse Conditioning Fix** → target_length: 256
3. **KSampler** → Verwende die "fixed_conditioning" Ausgabe

### **Methode 2: Empty Conditioning für Negative**

```
CLIP → ⭕ Flux LayerDiffuse Empty Conditioning → KSampler (negative)
```

**Für negative Prompts:**
1. **⭕ Flux LayerDiffuse Empty Conditioning**
2. **KSampler** → Verwende als negative conditioning

### **Methode 3: Troubleshooter verwenden**

1. **🩺 Flux LayerDiffuse Troubleshooter**
2. **issue_type**: "tensor_dimension_error"
3. **Folge den Anweisungen** in der Ausgabe

## 🚀 **Kompletter korrigierter Workflow:**

```
1. Load Diffusion Model (Flux)
2. Load LoRA (layerlora.safetensors, Strength: 1.0)
3. CLIP Text Encode (Positive) → "glass bottle, transparent"
4. 🔧 Flux LayerDiffuse Conditioning Fix → target_length: 256
5. ⭕ Flux LayerDiffuse Empty Conditioning → für negative
6. Empty Latent Image → 1024x1024
7. KSampler → Verwende fixed_conditioning
8. 🔧 Flux LayerDiffuse Standalone Loader
9. 🔍 Flux LayerDiffuse Decoder (Simple)
10. Save Image
```

## ⚙️ **Einstellungen:**

### **Conditioning Fix:**
- **target_length**: 256 (Standard für Flux)
- Falls 256 nicht funktioniert, versuche: 77, 512

### **Empty Conditioning:**
- **sequence_length**: 256 (gleich wie Conditioning Fix)
- **batch_size**: 1

### **KSampler:**
- **Steps**: 20-30
- **CFG**: 3.5-7.0
- **Sampler**: euler
- **Scheduler**: normal

## 🔍 **Debugging:**

### **Tensor-Shapes prüfen:**
Die Nodes zeigen in der Konsole:
```
Original conditioning shape: [1, 77, 2048]
Fixed conditioning shape: [1, 256, 4096]
```

### **Häufige Probleme:**

1. **Falsche target_length:**
   - Versuche 77, 256, oder 512
   - Schaue in Konsole nach "Fixed conditioning shape"

2. **CLIP-Modell inkompatibel:**
   - Verwende Flux-kompatible CLIP
   - Teste mit Empty Conditioning

3. **Batch-Size Probleme:**
   - Setze batch_size auf 1
   - Verwende gleiche Dimensionen überall

## 🎯 **Quick Fix Checklist:**

- [ ] **🔧 Conditioning Fix** zwischen CLIP und KSampler
- [ ] **⭕ Empty Conditioning** für negative prompts
- [ ] **target_length = 256** in beiden Nodes
- [ ] **batch_size = 1** überall
- [ ] **Flux-kompatible CLIP** verwenden
- [ ] **ComfyUI neu starten** nach Node-Installation

## 🆘 **Wenn es immer noch nicht funktioniert:**

1. **🩺 Troubleshooter** verwenden
2. **Konsole prüfen** für genaue Tensor-Shapes
3. **Verschiedene target_length** ausprobieren (77, 256, 512)
4. **Anderes CLIP-Modell** testen
5. **Einfacheren Prompt** verwenden ("glass bottle")

## 💡 **Warum passiert das?**

- **Flux-Modelle** erwarten andere Tensor-Dimensionen als Standard SD
- **CLIP-Encoding** muss für Flux angepasst werden
- **Sequence Length** muss zwischen CLIP und Model übereinstimmen
- **Hidden Dimensions** müssen kompatibel sein

## ✅ **Nach dem Fix:**

Du solltest sehen:
```
✓ Fixed conditioning shape: [1, 256, 4096]
✓ Created empty conditioning: [1, 256, 4096]
✓ Decoding latent shape: [1, 16, 128, 128]
✓ Decoded transparent image: [1, 1024, 1024, 4]
```

**Jetzt sollte die transparente Bildgenerierung funktionieren!** 🎨✨
