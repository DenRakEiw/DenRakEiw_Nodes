#!/usr/bin/env python3
"""
Debug script to test ComfyUI node loading
Run this from the ComfyUI root directory to simulate how ComfyUI loads the nodes
"""

import sys
import os
import traceback

# Add ComfyUI root to path (simulate ComfyUI environment)
comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, comfyui_root)

print(f"🔍 ComfyUI Root: {comfyui_root}")
print(f"🔍 Current Working Directory: {os.getcwd()}")
print(f"🔍 Python Path: {sys.path[:3]}...")

def test_node_loading():
    """Test loading nodes like ComfyUI would"""
    
    print("\n🚀 Testing ComfyUI Node Loading...")
    print("=" * 50)
    
    # Change to the node directory
    node_dir = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    
    try:
        os.chdir(node_dir)
        print(f"📁 Changed to node directory: {node_dir}")
        
        # Try to import the module like ComfyUI would
        print("\n📦 Importing node module...")
        
        # Add the custom_nodes directory to path
        custom_nodes_dir = os.path.dirname(node_dir)
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)
        
        # Import the module
        module_name = os.path.basename(node_dir)
        print(f"🔧 Importing module: {module_name}")
        
        # Try importing like ComfyUI does
        spec = None
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(node_dir, "__init__.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print("✅ Module imported successfully!")
            
            # Check for required attributes
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                mappings = module.NODE_CLASS_MAPPINGS
                print(f"✅ NODE_CLASS_MAPPINGS found: {len(mappings)} nodes")
                for node_name, node_class in mappings.items():
                    print(f"   - {node_name}: {node_class}")
                    
                    # Test if the class has required methods
                    if hasattr(node_class, 'INPUT_TYPES'):
                        try:
                            input_types = node_class.INPUT_TYPES()
                            print(f"     ✓ INPUT_TYPES: OK")
                        except Exception as e:
                            print(f"     ❌ INPUT_TYPES error: {e}")
                    else:
                        print(f"     ❌ Missing INPUT_TYPES method")
            else:
                print("❌ NODE_CLASS_MAPPINGS not found!")
                return False
            
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                display_mappings = module.NODE_DISPLAY_NAME_MAPPINGS
                print(f"✅ NODE_DISPLAY_NAME_MAPPINGS found: {len(display_mappings)} names")
                for node_name, display_name in display_mappings.items():
                    print(f"   - {node_name}: '{display_name}'")
            else:
                print("❌ NODE_DISPLAY_NAME_MAPPINGS not found!")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            traceback.print_exc()
            return False
            
    finally:
        os.chdir(original_cwd)

def check_dependencies():
    """Check if all dependencies are available"""
    
    print("\n🔍 Checking Dependencies...")
    print("-" * 30)
    
    dependencies = [
        "torch",
        "numpy", 
        "PIL",
        "diffusers",
        "transformers",
        "safetensors",
        "cv2",
        "tqdm"
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == "PIL":
                import PIL
            elif dep == "cv2":
                import cv2
            else:
                __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def main():
    print("🔧 ComfyUI Node Loading Debug Tool")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    # Test node loading
    load_ok = test_node_loading()
    
    print("\n" + "=" * 50)
    if deps_ok and load_ok:
        print("🎉 SUCCESS! Nodes should load correctly in ComfyUI")
        print("\nIf nodes still don't appear in ComfyUI:")
        print("1. Restart ComfyUI completely")
        print("2. Check ComfyUI console for error messages")
        print("3. Clear ComfyUI cache if available")
        print("4. Check that the custom_nodes folder is in the right location")
    else:
        print("❌ ISSUES FOUND! Please fix the errors above.")
        if not deps_ok:
            print("- Install missing dependencies")
        if not load_ok:
            print("- Fix import/loading errors")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
