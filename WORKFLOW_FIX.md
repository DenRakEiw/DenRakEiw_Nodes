# 🔧 Workflow Fix für Tensor Dimension Error

## ❌ **Dein aktuelles Problem:**

Du verwendest:
```
CLIPTextEncode → KSampler (positive)  ← HIER ist das Problem!
CLIPTextEncode → ConditioningZeroOut → KSampler (negative)  ← Das ist OK
```

Der **Tensor Dimension Error** kommt vom **positiven** Conditioning, nicht vom negativen!

## ✅ **Korrekte Lösung:**

### **Schritt 1: Positive Conditioning fixen**

```
CLIPTextEncode → 🔧 Flux LayerDiffuse Conditioning Fix → KSampler (positive)
```

### **Schritt 2: Negative Conditioning (bleibt wie es ist)**

```
CLIPTextEncode → ConditioningZeroOut → KSampler (negative)
```

## 🚀 **Korrigierter Workflow:**

1. **DualCLIPLoader** → CLIP laden
2. **CLIPTextEncode** → "a football, transparent"
3. **🔧 Flux LayerDiffuse Conditioning Fix** → target_length: 256
4. **ConditioningZeroOut** → für negative (wie du es schon hast)
5. **KSampler** → 
   - positive: **fixed_conditioning** (von Conditioning Fix)
   - negative: **CONDITIONING** (von ConditioningZeroOut)

## 🔧 **Was du ändern musst:**

### **Neue Verbindung:**
```
CLIPTextEncode → Flux LayerDiffuse Conditioning Fix → KSampler (positive)
```

### **Alte Verbindung entfernen:**
```
❌ CLIPTextEncode → KSampler (positive)  # Diese Verbindung löschen!
```

## ⚙️ **Node-Einstellungen:**

### **Flux LayerDiffuse Conditioning Fix:**
- **conditioning**: Von CLIPTextEncode
- **target_length**: 256

### **KSampler:**
- **positive**: Von **Conditioning Fix** (nicht direkt von CLIPTextEncode!)
- **negative**: Von ConditioningZeroOut (bleibt gleich)

## 🎯 **Warum passiert das?**

- **CLIPTextEncode** gibt Tensors mit Shape `[1, 77, 2048]` aus
- **Flux** erwartet aber `[1, 256, 4096]`
- **Conditioning Fix** konvertiert: `77x2048 → 256x4096`
- **ConditioningZeroOut** ist OK, weil es die Dimensionen beibehält

## 📋 **Schnelle Checkliste:**

- [ ] **🔧 Conditioning Fix** Node hinzufügen
- [ ] **CLIPTextEncode** → **Conditioning Fix** verbinden
- [ ] **Conditioning Fix** → **KSampler (positive)** verbinden
- [ ] **Alte Verbindung** CLIPTextEncode → KSampler löschen
- [ ] **target_length = 256** einstellen
- [ ] **ConditioningZeroOut** bleibt unverändert

## ✅ **Nach dem Fix solltest du sehen:**

```
Original conditioning shape: [1, 77, 2048]
Fixed conditioning shape: [1, 256, 4096]
```

**Dann sollte der Tensor Dimension Error verschwinden!** 🎨✨
