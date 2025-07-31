# 🔧 Korrigierter Flux LayerDiffuse Workflow

## ❌ **Das Problem war:**
Die TransparentVAE braucht eine **echte VAE** zum Dekodieren der Latents, nicht nur eine Dummy-VAE.

## ✅ **Korrekte Lösung:**

### **Flux LayerDiffuse Standalone Loader braucht jetzt VAE Input:**
```
VAELoader → Flux LayerDiffuse Standalone Loader
```

### **Decoder braucht KEINE separate VAE mehr:**
```
Flux LayerDiffuse Standalone Loader → Flux LayerDiffuse Decoder (Simple)
KSampler → Flux LayerDiffuse Decoder (Simple)
```

## 🚀 **Finaler korrekter Workflow:**

```
1. UNETLoader → Flux Model
2. LoraLoaderModelOnly → layerlora.safetensors (Strength: 1.0)
3. DualCLIPLoader → type: "flux"
4. CLIPTextEncode → "a football, transparent"
5. ConditioningZeroOut → für negative
6. EmptySD3LatentImage → 1024x1024
7. KSampler → Standard Sampling
8. VAELoader → Standard VAE
9. Flux LayerDiffuse Standalone Loader → TransparentVAE + VAE
10. Flux LayerDiffuse Decoder (Simple) → NUR transparent_vae + samples
11. Save Image
```

## 🔗 **Neue Verbindungen:**

### **Model Loading:**
```
UNETLoader → LoraLoaderModelOnly → KSampler (model)
DualCLIPLoader → CLIPTextEncode → KSampler (positive)
DualCLIPLoader → CLIPTextEncode → ConditioningZeroOut → KSampler (negative)
EmptySD3LatentImage → KSampler (latent_image)
```

### **VAE und TransparentVAE:**
```
VAELoader → Flux LayerDiffuse Standalone Loader (vae)
Flux LayerDiffuse Standalone Loader → Flux LayerDiffuse Decoder (transparent_vae)
```

### **Decoding:**
```
KSampler (latent) → Flux LayerDiffuse Decoder (samples)
```

## ⚙️ **Wichtige Änderungen:**

### **Flux LayerDiffuse Standalone Loader:**
- **JETZT**: Braucht VAE Input
- **transparent_vae_checkpoint**: "TransparentVAE.pth"
- **alpha**: 300.0
- **latent_channels**: 16
- **dtype**: bfloat16

### **Flux LayerDiffuse Decoder (Simple):**
- **JETZT**: Braucht KEINE separate VAE mehr
- **transparent_vae**: Von Standalone Loader
- **samples**: Von KSampler
- **use_augmentation**: true

## 📋 **Was sich geändert hat:**

### ✅ **Neu hinzugefügt:**
- **VAELoader** → **Flux LayerDiffuse Standalone Loader**

### ❌ **Entfernt:**
- **VAE Input** im Decoder (nicht mehr nötig)
- **Conditioning Fix** (mit type="flux" überflüssig)

### 🔄 **Gleich geblieben:**
- **DualCLIPLoader type="flux"**
- **KSampler** Standard-Einstellungen
- **ConditioningZeroOut** für negative

## 🎯 **Warum diese Änderung:**

1. **TransparentVAE.decode()** macht intern `self.sd_vae.decode(latent)`
2. **sd_vae** muss eine echte VAE sein, die Latents dekodieren kann
3. **MinimalVAE** konnte das nicht → Fehler
4. **Echte VAE** vom VAELoader kann das → Funktioniert!

## 🔍 **Console Output sollte zeigen:**

```
Input latent shape: [1, 16, 128, 128]
Latent device: cuda:0, dtype: torch.bfloat16
TransparentVAE device: cuda:0
Decoding with TransparentVAE (includes internal VAE decode)...
✓ Decoded transparent image: [1, 1024, 1024, 4]
```

## 🎉 **Jetzt sollte es funktionieren!**

**Starte ComfyUI neu und verwende den korrigierten Workflow mit VAE Input im Standalone Loader!** 🎨✨
