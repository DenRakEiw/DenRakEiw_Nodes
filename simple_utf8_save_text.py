import os


class UTF8CaptionSaver:
    """
    A ComfyUI node that saves text captions to UTF-8 encoded .txt files.
    Perfect for AI training datasets - clean, simple, no metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption_text": ("STRING", {
                    "multiline": True,
                    "default": "Enter caption text here...",
                    "placeholder": "Caption for AI training"
                }),
                "folder_path": ("STRING", {
                    "default": "output/captions",
                    "placeholder": "Folder path for caption files"
                }),
                "filename": ("STRING", {
                    "default": "caption",
                    "placeholder": "Filename (without .txt)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_caption"
    CATEGORY = "denrakeiw/text"
    OUTPUT_NODE = True

    def save_caption(self, caption_text, folder_path, filename):
        """Save caption text to UTF-8 .txt file for AI training"""
        print(f"[UTF8CaptionSaver] Saving caption...")
        print(f"[UTF8CaptionSaver] Caption length: {len(caption_text)} characters")
        print(f"[UTF8CaptionSaver] Folder: {folder_path}")
        print(f"[UTF8CaptionSaver] Filename: {filename}")

        try:
            # Clean filename (remove invalid characters)
            clean_filename = filename.replace(" ", "_")
            for char in '<>:"/\\|?*':
                clean_filename = clean_filename.replace(char, "_")

            # Create final filename (no timestamp for AI training)
            final_filename = f"{clean_filename}.txt"
            
            # Create full path
            if os.path.isabs(folder_path):
                full_folder = folder_path
            else:
                full_folder = os.path.join("output", folder_path)
            
            # Create directory
            os.makedirs(full_folder, exist_ok=True)
            print(f"[UTF8CaptionSaver] Created directory: {full_folder}")

            # Full file path
            file_path = os.path.join(full_folder, final_filename)
            print(f"[UTF8CaptionSaver] Full file path: {file_path}")

            # Write caption file with UTF-8 encoding (clean, no metadata)
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(caption_text)

            print(f"[UTF8CaptionSaver] ✓ Caption saved: {file_path}")
            return (file_path,)
            
        except Exception as e:
            error_msg = f"✗ Error: {str(e)}"
            print(f"[UTF8CaptionSaver] {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)
