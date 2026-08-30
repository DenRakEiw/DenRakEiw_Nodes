# 🔒 SAFETY CHECKLIST FOR TRAINING

## 🚨 CRITICAL CHECKS BEFORE TRAINING

### ✅ **1. VAE AUTHENTICITY (THE MOST IMPORTANT CHECK!)**

**🔍 Only ever use these VAEs:**
- ✅ `stabilityai/sd-vae-ft-mse` (RECOMMENDED)
- ✅ `stabilityai/sd-vae-ft-ema` (RECOMMENDED)

**❌ NEVER use:**
- ❌ Unknown VAE models
- ❌ Custom or modified VAEs
- ❌ VAEs from unverified sources
- ❌ Local VAE files that have not been verified

**🔍 Automatic check:**
```bash
python validation_system.py
```

### ✅ **2. LATENT VALIDATION**

**What a correct latent looks like:**
- ✅ Shape: `[4, H, W]` (4 channels)
- ✅ Value range: roughly [-5, +5]
- ✅ No NaN or Inf values
- ✅ Consistent dimensions

**How to spot a bad latent:**
- ❌ Wrong channel count (not 4)
- ❌ Extreme values (>10 or <-10)
- ❌ Corrupt files
- ❌ Inconsistent sizes

### ✅ **3. DATASET QUALITY**

**Before/after pairs:**
- ✅ 32x32 → 64x64 latents
- ✅ Same content at different resolutions
- ✅ Correct interpolation
- ✅ No artefacts

**Dataset size:**
- ✅ Minimum: 500 training samples
- ✅ Recommended: 1000+ training samples
- ✅ Validation: 10% of the training size

### ✅ **4. SYSTEM REQUIREMENTS**

**Hardware:**
- ✅ GPU: 6 GB+ VRAM (8 GB+ recommended)
- ✅ RAM: 16 GB+ (32 GB recommended)
- ✅ Disk: 15 GB+ free

**Software:**
- ✅ PyTorch with CUDA
- ✅ The diffusers library
- ✅ An internet connection for downloads

## 🔍 AUTOMATIC VALIDATION

### **Full check:**
```bash
python pre_training_checker.py
```

### **Check the VAE only:**
```bash
python validation_system.py
```

### **Check the datasets only:**
```bash
python dataset_preparation.py
```

## 🚨 WARNING SIGNS

### **❌ STOP IMMEDIATELY when:**
- A fake VAE is detected
- Corrupt latents are found
- The data contains extreme values
- Memory or GPU errors appear
- The internet connection drops

### **⚠️ BE CAREFUL when:**
- There are few training samples (<500)
- VRAM is low (<6 GB)
- Latent values look unusual
- The internet connection is slow

## 🎯 TRAINING SAFETY

### **During training:**
- ✅ Save checkpoints regularly
- ✅ Monitor the loss
- ✅ Run validation checks
- ✅ Watch the GPU temperature

### **After training:**
- ✅ Validate the model
- ✅ Run a test inference
- ✅ Compare the quality
- ✅ Make a backup

## 🔧 TROUBLESHOOTING

### **VAE problems:**
```bash
# Check that the VAE is genuine
python -c "from validation_system import VAEValidator; VAEValidator().validate()"

# Load only an official VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

### **Latent problems:**
```bash
# Check the latent files
python -c "from validation_system import LatentValidator; LatentValidator().validate_latent_directory('datasets/latents/train')"

# Rebuild them if something is wrong
rm -rf datasets/latents/
python dataset_preparation.py
```

### **Dataset problems:**
```bash
# Rebuild everything
rm -rf datasets/
python dataset_preparation.py
```

### **GPU problems:**
```bash
# Check the VRAM
nvidia-smi

# Reduce the batch size
# In the config: batch_size = 8 (instead of 16)
```

## 📊 QUALITY CONTROL

### **Before training:**
1. ✅ Every validation passed
2. ✅ Visual tests look right
3. ✅ Sample latents decode correctly
4. ✅ System check OK

### **During training:**
1. ✅ The loss keeps falling
2. ✅ No NaN or Inf values
3. ✅ GPU utilisation is stable
4. ✅ Checkpoints are being written

### **After training:**
1. ✅ The model loads correctly
2. ✅ Test inference works
3. ✅ Quality beats the baseline
4. ✅ No artefacts

## 🎉 SUCCESS CRITERIA

### **Training worked when:**
- ✅ Validation loss < training loss
- ✅ Visual quality improved
- ✅ No artefacts in the outputs
- ✅ The model converged stably
- ✅ Test samples look good

### **Repeat the training when:**
- ❌ The loss keeps rising
- ❌ There are strong artefacts
- ❌ The model diverges
- ❌ Quality is worse than the baseline
- ❌ NaN or Inf values appear

## 🚀 FINAL CHECKLIST

**Before training:**
- [ ] VAE authenticity checked
- [ ] Latents validated
- [ ] Dataset quality confirmed
- [ ] System requirements met
- [ ] Every validation passed

**Start the training:**
```bash
# Recommended order:
python pre_training_checker.py  # full check
python train_advanced_upscaler.py  # start training
```

**If something goes wrong:**
```bash
# Debug mode
python validation_system.py  # detailed check
python dataset_preparation.py  # rebuild
```

---

## 🔒 **THE MOST IMPORTANT RULE:**

**NEVER START TRAINING WITHOUT A FULL VALIDATION!**

Validation prevents:
- ❌ Using a fake VAE
- ❌ Corrupt data
- ❌ Wasted training time
- ❌ Poor results
- ❌ System crashes

**Always run first:** `python pre_training_checker.py` 🔍
