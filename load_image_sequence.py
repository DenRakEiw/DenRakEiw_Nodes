"""
Load Image Sequence Node for ComfyUI
Loads images from a folder one by one, incrementing through the sequence
"""

import os
import torch
import numpy as np
from PIL import Image
import json
import hashlib

# ComfyUI imports - only available when running in ComfyUI
try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


class LoadImageSequence:
    """
    Load images from a folder one by one, incrementing through the sequence
    """
    
    # Class variable to store state across executions
    _sequence_states = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "reset_sequence": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "file_extensions": ("STRING", {
                    "default": "jpg,jpeg,png,bmp,tiff,webp",
                    "multiline": False
                }),
                "sort_method": (["alphabetical", "date_modified", "date_created", "file_size"], {
                    "default": "alphabetical"
                }),
                "loop_sequence": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "txt_filename", "current_index", "total_images")
    FUNCTION = "load_next_image"
    CATEGORY = "denrakeiw/image"

    def load_next_image(self, folder_path, reset_sequence, file_extensions="jpg,jpeg,png,bmp,tiff,webp", sort_method="alphabetical", loop_sequence=True):
        """
        Load the next image in the sequence from the specified folder
        """
        
        try:
            # Validate folder path
            if not folder_path or not os.path.exists(folder_path):
                raise ValueError(f"Folder path does not exist: {folder_path}")
            
            if not os.path.isdir(folder_path):
                raise ValueError(f"Path is not a directory: {folder_path}")
            
            # Create unique key for this folder path
            folder_key = hashlib.md5(folder_path.encode()).hexdigest()
            
            # Get list of supported image files
            extensions = [ext.strip().lower() for ext in file_extensions.split(',')]
            image_files = []
            
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    file_ext = os.path.splitext(file)[1][1:].lower()  # Remove dot and lowercase
                    if file_ext in extensions:
                        image_files.append(file)
            
            if not image_files:
                raise ValueError(f"No image files found in folder: {folder_path}")
            
            # Sort files based on method
            if sort_method == "alphabetical":
                image_files.sort()
            elif sort_method == "date_modified":
                image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
            elif sort_method == "date_created":
                image_files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
            elif sort_method == "file_size":
                image_files.sort(key=lambda x: os.path.getsize(os.path.join(folder_path, x)))
            
            # Initialize or get current state
            if folder_key not in self._sequence_states or reset_sequence:
                self._sequence_states[folder_key] = {
                    'current_index': 0,
                    'image_files': image_files,
                    'folder_path': folder_path,
                    'last_execution': 0
                }
                print(f"✓ Initialized sequence for folder: {folder_path}")
                print(f"  Found {len(image_files)} images")
            else:
                # Update image list in case files changed
                old_files = self._sequence_states[folder_key]['image_files']
                if old_files != image_files:
                    print(f"✓ Updated image list for folder: {folder_path}")
                    print(f"  Images changed: {len(old_files)} → {len(image_files)}")
                    # Reset index if current index is out of bounds
                    if self._sequence_states[folder_key]['current_index'] >= len(image_files):
                        self._sequence_states[folder_key]['current_index'] = 0
                    self._sequence_states[folder_key]['image_files'] = image_files
            
            state = self._sequence_states[folder_key]

            # Check if this is a new execution (prevent multiple increments in same workflow)
            import time
            current_time = time.time()
            if current_time - state.get('last_execution', 0) > 0.1:  # 100ms threshold
                # This is a new execution, use current index and then increment
                current_index = state['current_index']
                state['last_execution'] = current_time

                # Increment for next time AFTER we use current index
                if loop_sequence:
                    state['current_index'] = (current_index + 1) % len(image_files)
                else:
                    state['current_index'] = min(current_index + 1, len(image_files) - 1)
            else:
                # Same execution, don't increment again
                current_index = state['current_index']
            
            # Handle index bounds
            if current_index >= len(image_files):
                if loop_sequence:
                    current_index = 0
                    state['current_index'] = 0
                    print("✓ Sequence looped back to start")
                else:
                    current_index = len(image_files) - 1
                    state['current_index'] = current_index
                    print("✓ Sequence at end, staying on last image")
            
            # Get current image file
            current_file = image_files[current_index]
            image_path = os.path.join(folder_path, current_file)
            
            print(f"✓ Loading image {current_index + 1}/{len(image_files)}: {current_file}")
            
            # Load and process image
            try:
                pil_image = Image.open(image_path)
                
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                
                # Convert to tensor [1, H, W, C]
                image_tensor = torch.from_numpy(image_np).unsqueeze(0)

                # Generate txt filename (replace extension with .txt)
                txt_filename = os.path.splitext(current_file)[0] + ".txt"

                print(f"✓ Loaded image {current_index + 1}/{len(image_files)}: {current_file}")
                print(f"  Corresponding txt file: {txt_filename}")
                print(f"  Next image will be index: {state['current_index']}")

                return (
                    image_tensor,
                    current_file,
                    txt_filename,
                    current_index + 1,  # 1-based for user display
                    len(image_files)
                )
                
            except Exception as e:
                print(f"✗ Error loading image {current_file}: {str(e)}")
                # Image was already incremented, no need to increment again
                
                # Create a placeholder image
                placeholder = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                error_txt_filename = os.path.splitext(current_file)[0] + ".txt" if current_file else "error.txt"
                return (
                    placeholder,
                    f"ERROR: {current_file}",
                    error_txt_filename,
                    current_index + 1,
                    len(image_files)
                )
            
        except Exception as e:
            print(f"✗ Error in LoadImageSequence: {str(e)}")
            
            # Return a placeholder image on error
            placeholder = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
            return (
                placeholder,
                f"ERROR: {str(e)}",
                "error.txt",
                0,
                0
            )

    @classmethod
    def IS_CHANGED(cls, folder_path, reset_sequence, **kwargs):
        """
        Check if the node should be re-executed
        Always return a unique value to force re-execution for sequence progression
        """
        import time
        import random

        # Always return a unique value to force re-execution
        # This ensures the sequence progresses with each workflow run
        unique_id = f"{time.time()}_{random.random()}"

        # Also check if reset is requested
        if reset_sequence:
            unique_id += "_reset"

        return unique_id


class LoadImageSequenceInfo:
    """
    Display information about the current image sequence
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_sequence_info"
    CATEGORY = "denrakeiw/image"

    def get_sequence_info(self, folder_path):
        """
        Get information about the image sequence
        """
        
        try:
            if not folder_path or not os.path.exists(folder_path):
                return (f"Folder does not exist: {folder_path}",)
            
            # Count image files
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
            image_files = []
            
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    file_ext = os.path.splitext(file)[1][1:].lower()
                    if file_ext in extensions:
                        image_files.append(file)
            
            image_files.sort()
            
            # Get current state
            folder_key = hashlib.md5(folder_path.encode()).hexdigest()
            current_index = 0
            if folder_key in LoadImageSequence._sequence_states:
                current_index = LoadImageSequence._sequence_states[folder_key]['current_index']
            
            info_lines = [
                "=== Image Sequence Info ===",
                f"Folder: {folder_path}",
                f"Total images: {len(image_files)}",
                f"Current index: {current_index + 1}/{len(image_files)}" if image_files else "Current index: 0/0",
                f"Next image: {image_files[current_index] if current_index < len(image_files) else 'None'}",
                "",
                "Recent files:"
            ]
            
            # Show first 10 files
            for i, file in enumerate(image_files[:10]):
                marker = "→ " if i == current_index else "  "
                info_lines.append(f"{marker}{i+1:2d}. {file}")
            
            if len(image_files) > 10:
                info_lines.append(f"  ... and {len(image_files) - 10} more files")
            
            return ("\n".join(info_lines),)
            
        except Exception as e:
            return (f"Error getting sequence info: {str(e)}",)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadImageSequence": LoadImageSequence,
    "LoadImageSequenceInfo": LoadImageSequenceInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageSequence": "📁 Load Image Sequence",
    "LoadImageSequenceInfo": "📊 Load Image Sequence Info",
}
