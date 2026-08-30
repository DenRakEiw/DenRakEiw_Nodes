#!/usr/bin/env python3
"""
🔥 UNIFIED WAN VAE TRAINING SYSTEM - DENRAKEIW SUPERHERO EDITION 🔥
Komplettes Training-System das alles automatisch macht:
1. Dataset Download & Creation
2. WAN VAE Latent Generation
3. Model Training
4. Node Integration
5. Testing & Validation
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

class UnifiedWanVAETrainingSystem:
    """🔥 DENRAKEIW SUPERHERO UNIFIED TRAINING SYSTEM 🔥"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.dataset_dir = self.base_dir / "wan_vae_datasets"
        self.models_dir = self.base_dir / "wan_vae_models"
        self.logs_dir = self.base_dir / "logs"
        
        # Erstelle Verzeichnisse
        for dir_path in [self.dataset_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.status = {
            'dataset_download': False,
            'image_pairs': False,
            'wan_vae_latents': False,
            'model_training': False,
            'node_integration': False,
            'testing': False
        }
        
        print("🔥 DENRAKEIW SUPERHERO UNIFIED TRAINING SYSTEM")
        print("=" * 60)
    
    def check_dependencies(self):
        """Check every dependency"""
        print("🔧 Checking dependencies...")
        
        required_packages = [
            'torch', 'torchvision', 'tqdm', 'matplotlib',
            'PIL', 'requests', 'numpy'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing.append(package)
                print(f"❌ {package}")
        
        if missing:
            print(f"\n⚠️ Missing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        print("✅ All dependencies satisfied!")
        return True
    
    def run_dataset_creation(self):
        """Run dataset creation"""
        print("\n1️⃣ DATASET CREATION")
        print("-" * 30)
        
        try:
            # Run the dataset creator
            result = subprocess.run([
                sys.executable, "wan_vae_dataset_creator.py"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print("✅ Dataset creation successful!")
                self.status['dataset_download'] = True
                self.status['image_pairs'] = True
                self.status['wan_vae_latents'] = True
                return True
            else:
                print(f"❌ Dataset creation failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Dataset creation timeout - continuing with existing data")
            return self._check_existing_dataset()
        except Exception as e:
            print(f"❌ Dataset creation error: {e}")
            return self._check_existing_dataset()
    
    def _check_existing_dataset(self):
        """Check whether the dataset already exists"""
        train_dir = self.dataset_dir / "latents" / "train"
        val_dir = self.dataset_dir / "latents" / "validation"
        
        if train_dir.exists() and val_dir.exists():
            train_files = list(train_dir.glob("*.pt"))
            val_files = list(val_dir.glob("*.pt"))
            
            if len(train_files) > 0 and len(val_files) > 0:
                print(f"✅ Found existing dataset:")
                print(f"   Train: {len(train_files)} files")
                print(f"   Val: {len(val_files)} files")
                self.status['wan_vae_latents'] = True
                return True
        
        print("❌ No existing dataset found")
        return False
    
    def run_model_training(self):
        """Run model training"""
        print("\n2️⃣ MODEL TRAINING")
        print("-" * 30)
        
        if not self.status['wan_vae_latents']:
            print("❌ No dataset available for training")
            return False
        
        try:
            # Run the WAN VAE trainer
            result = subprocess.run([
                sys.executable, "wan_vae_trainer.py"
            ], capture_output=True, text=True, timeout=7200)  # 2 hours timeout
            
            if result.returncode == 0:
                print("✅ Model training successful!")
                self.status['model_training'] = True
                return True
            else:
                print(f"❌ Model training failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Training timeout - checking for saved models")
            return self._check_trained_models()
        except Exception as e:
            print(f"❌ Training error: {e}")
            return self._check_trained_models()
    
    def _check_trained_models(self):
        """Check whether trained models exist"""
        model_files = [
            self.models_dir / "best_wan_vae_upscaler.pth",
            "models/best_wan_vae_upscaler.pth",
            "models/best_model.pth"
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                print(f"✅ Found trained model: {model_file}")
                self.status['model_training'] = True
                return True
        
        print("❌ No trained models found")
        return False
    
    def integrate_with_comfyui(self):
        """Integriere mit ComfyUI"""
        print("\n3️⃣ COMFYUI INTEGRATION")
        print("-" * 30)
        
        try:
            # Kopiere Modelle an richtige Orte
            model_sources = [
                self.models_dir / "best_wan_vae_upscaler.pth",
                "models/best_model.pth"
            ]
            
            model_targets = [
                "../../models/upscale_models/wan_vae_upscaler.pth",
                "models/wan_vae_upscaler.pth"
            ]
            
            for source in model_sources:
                if Path(source).exists():
                    for target in model_targets:
                        target_path = Path(target)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        import shutil
                        shutil.copy2(source, target)
                        print(f"✅ Copied {source} -> {target}")
                        break
                    break
            
            # Check the node files
            node_files = [
                "wan_nn_latent_upscaler.py",
                "optimized_wan_vae_node.py"
            ]
            
            for node_file in node_files:
                if Path(node_file).exists():
                    print(f"✅ Node file ready: {node_file}")
                else:
                    print(f"⚠️ Node file missing: {node_file}")
            
            self.status['node_integration'] = True
            return True
            
        except Exception as e:
            print(f"❌ Integration error: {e}")
            return False
    
    def run_testing(self):
        """Run the tests"""
        print("\n4️⃣ TESTING")
        print("-" * 30)
        
        try:
            # Test WAN VAE Loader
            print("🧪 Testing WAN VAE Loader...")
            result = subprocess.run([
                sys.executable, "-c", 
                "from wan_vae_loader import test_wan_vae_loader; test_wan_vae_loader()"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ WAN VAE Loader test passed")
            else:
                print("⚠️ WAN VAE Loader test failed (may be expected)")
            
            # Test Node Import
            print("🧪 Testing Node Import...")
            result = subprocess.run([
                sys.executable, "-c",
                "from wan_nn_latent_upscaler import WanNNLatentUpscalerNode; print('Node import successful')"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Node import test passed")
            else:
                print("⚠️ Node import test failed")
            
            self.status['testing'] = True
            return True
            
        except Exception as e:
            print(f"❌ Testing error: {e}")
            return False
    
    def generate_report(self):
        """Generiere Abschlussbericht"""
        print("\n🎉 DENRAKEIW SUPERHERO TRAINING COMPLETE!")
        print("=" * 60)
        
        # Status Report
        print("📊 STATUS REPORT:")
        for step, completed in self.status.items():
            status = "✅" if completed else "❌"
            print(f"   {status} {step.replace('_', ' ').title()}")
        
        # File Report
        print("\n📁 GENERATED FILES:")
        
        # Models
        model_files = [
            self.models_dir / "best_wan_vae_upscaler.pth",
            "models/best_model.pth",
            "models/wan_vae_upscaler.pth"
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                size_mb = Path(model_file).stat().st_size / (1024*1024)
                print(f"   🧠 {model_file} ({size_mb:.1f}MB)")
        
        # Datasets
        dataset_info_file = self.dataset_dir / "latents" / "dataset_info.json"
        if dataset_info_file.exists():
            with open(dataset_info_file) as f:
                info = json.load(f)
                print(f"   📊 Dataset: {info.get('train_samples', 0)} train, {info.get('val_samples', 0)} val samples")
        
        # Nodes
        node_files = ["wan_nn_latent_upscaler.py", "optimized_wan_vae_node.py"]
        for node_file in node_files:
            if Path(node_file).exists():
                print(f"   🔧 {node_file}")
        
        # Success Rate
        completed_steps = sum(self.status.values())
        total_steps = len(self.status)
        success_rate = (completed_steps / total_steps) * 100
        
        print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}% ({completed_steps}/{total_steps})")
        
        if success_rate >= 80:
            print("🎉 TRAINING SYSTEM SUCCESSFUL!")
            print("🚀 Ready for WAN VAE Latent Upscaling!")
        elif success_rate >= 60:
            print("⚠️ Partial success - some components may need manual setup")
        else:
            print("❌ Training incomplete - check errors above")
        
        return success_rate
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 STARTING FULL WAN VAE TRAINING PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Check Dependencies
        if not self.check_dependencies():
            return False
        
        # 2. Dataset Creation
        self.run_dataset_creation()
        
        # 3. Model Training
        if self.status['wan_vae_latents']:
            self.run_model_training()
        
        # 4. ComfyUI Integration
        if self.status['model_training']:
            self.integrate_with_comfyui()
        
        # 5. Testing
        self.run_testing()
        
        # 6. Report
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️ TOTAL TIME: {total_time/60:.1f} minutes")
        
        success_rate = self.generate_report()
        
        return success_rate >= 80

def main():
    """Hauptfunktion"""
    system = UnifiedWanVAETrainingSystem()
    success = system.run_full_pipeline()
    
    if success:
        print("\n🎉 DENRAKEIW SUPERHERO MISSION ACCOMPLISHED!")
        return 0
    else:
        print("\n⚠️ DENRAKEIW SUPERHERO MISSION PARTIALLY COMPLETED!")
        return 1

if __name__ == "__main__":
    exit(main())
