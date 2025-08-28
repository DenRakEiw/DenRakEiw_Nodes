#!/usr/bin/env python3
"""
Test WAN VAE Node - DENRAKEIW SUPERHERO EDITION
Standalone test for the WAN VAE upscaler functionality
"""

import torch
import os
import sys

# Mock ComfyUI modules for testing
class MockModelManagement:
    @staticmethod
    def get_torch_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add mock modules to sys.modules
sys.modules['comfy.model_management'] = MockModelManagement()
sys.modules['folder_paths'] = type('MockFolderPaths', (), {})()

# Now import our node
from simple_wan_vae_training import SimpleWanVAEUpscaler

class TestWanVAEUpscaler:
    """Test class for WAN VAE Upscaler"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        print(f"DENRAKEIW SUPERHERO WAN VAE TESTER")
        print(f"Device: {self.device}")
    
    def load_model(self):
        """Load the trained WAN VAE model"""
        model_paths = [
            "models/simple_wan_vae_upscaler.pth",
            "custom_nodes/denrakeiw_nodes/models/simple_wan_vae_upscaler.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("ERROR: No WAN VAE model found!")
            return False
        
        try:
            print(f"Loading WAN VAE model: {model_path}")
            
            # Create model
            self.model = SimpleWanVAEUpscaler(input_channels=16, output_channels=16)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', 'unknown')
                print(f"Model loaded from epoch {epoch} with val_loss {val_loss}")
            else:
                self.model.load_state_dict(checkpoint)
                print("Model state dict loaded")
            
            self.model.to(self.device)
            self.model.eval()
            
            print("WAN VAE model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_upscaling(self):
        """Test the upscaling functionality"""
        if self.model is None:
            print("ERROR: No model loaded!")
            return False
        
        print("\nTesting WAN VAE upscaling...")
        
        # Create test latent (simulating WAN VAE latent)
        test_latent = {
            "samples": torch.randn(2, 16, 32, 32)  # Batch of 2, 16 channels, 32x32
        }
        
        print(f"Input latent shape: {test_latent['samples'].shape}")
        
        try:
            # Simulate the node's upscale function
            input_tensor = test_latent["samples"].to(self.device)
            
            with torch.no_grad():
                upscaled = self.model(input_tensor)
            
            upscaled = upscaled.cpu()
            
            print(f"Output latent shape: {upscaled.shape}")
            
            # Validate output
            expected_shape = (2, 16, 64, 64)
            if upscaled.shape == expected_shape:
                print("SUCCESS: Output shape is correct!")
                
                # Check if values are reasonable
                mean_val = upscaled.mean().item()
                std_val = upscaled.std().item()
                print(f"Output statistics: mean={mean_val:.4f}, std={std_val:.4f}")
                
                if abs(mean_val) < 1.0 and 0.1 < std_val < 2.0:
                    print("SUCCESS: Output values are in reasonable range!")
                    return True
                else:
                    print("WARNING: Output values might be unusual")
                    return True
            else:
                print(f"ERROR: Wrong output shape! Expected {expected_shape}, got {upscaled.shape}")
                return False
                
        except Exception as e:
            print(f"Upscaling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_different_inputs(self):
        """Test with different input configurations"""
        if self.model is None:
            print("ERROR: No model loaded!")
            return False
        
        print("\nTesting different input configurations...")
        
        test_cases = [
            (1, 16, 32, 32),   # Single sample
            (4, 16, 32, 32),   # Batch of 4
            (1, 16, 64, 64),   # Different input size (should still work)
        ]
        
        success_count = 0
        
        for i, shape in enumerate(test_cases):
            print(f"\nTest case {i+1}: {shape}")
            
            try:
                test_input = torch.randn(*shape).to(self.device)
                
                with torch.no_grad():
                    output = self.model(test_input)
                
                expected_h = shape[2] * 2  # 2x upscale
                expected_w = shape[3] * 2
                expected_shape = (shape[0], shape[1], expected_h, expected_w)
                
                if output.shape == expected_shape:
                    print(f"  SUCCESS: {shape} -> {output.shape}")
                    success_count += 1
                else:
                    print(f"  ERROR: Expected {expected_shape}, got {output.shape}")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\nDifferent inputs test: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)
    
    def run_all_tests(self):
        """Run all tests"""
        print("DENRAKEIW SUPERHERO WAN VAE TESTING")
        print("=" * 50)
        
        # Load model
        if not self.load_model():
            print("FAILED: Could not load model")
            return False
        
        # Test basic upscaling
        if not self.test_upscaling():
            print("FAILED: Basic upscaling test")
            return False
        
        # Test different inputs
        if not self.test_different_inputs():
            print("FAILED: Different inputs test")
            return False
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("WAN VAE UPSCALER IS READY FOR COMFYUI!")
        print("=" * 50)
        
        return True

def main():
    """Main test function"""
    tester = TestWanVAEUpscaler()
    success = tester.run_all_tests()
    
    if success:
        print("\nDENRAKEIW SUPERHERO WAN VAE TEST: SUCCESS!")
        return 0
    else:
        print("\nDENRAKEIW SUPERHERO WAN VAE TEST: FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
