import os

import folder_paths


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
                    "default": "captions",
                    "placeholder": "Folder inside the output directory"
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

        output_dir = os.path.realpath(folder_paths.get_output_directory())
        if os.path.isabs(folder_path) or os.path.splitdrive(folder_path)[0] or folder_path.startswith(("/", "\\")) or ".." in folder_path.replace("\\", "/").split("/"):
            raise ValueError(f"folder_path must be a relative folder inside the output directory: {folder_path}")

        try:
            # Clean filename (remove invalid characters)
            clean_filename = filename.replace(" ", "_")
            for char in '<>:"/\\|?*':
                clean_filename = clean_filename.replace(char, "_")

            # Create final filename (no timestamp for AI training)
            final_filename = f"{clean_filename}.txt"
            
            full_folder = os.path.join(output_dir, folder_path)
            file_path = os.path.realpath(os.path.join(full_folder, final_filename))
            if os.path.commonpath([output_dir, file_path]) != output_dir:
                raise ValueError(f"folder_path points outside the output directory: {folder_path}")

            os.makedirs(full_folder, exist_ok=True)
            print(f"[UTF8CaptionSaver] Created directory: {full_folder}")
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
