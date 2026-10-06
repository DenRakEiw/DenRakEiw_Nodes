import os
from urllib.parse import urlparse

import folder_paths
from comfy_api.input_impl import VideoFromFile
from yt_dlp import YoutubeDL


class InstagramVideoDownloader:
    """
    Downloads a video from an Instagram post, reel or story URL and returns it
    as a VIDEO so it can go straight into Save Video or Get Video Components.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "placeholder": "https://www.instagram.com/reel/..."
                }),
                "subfolder": ("STRING", {
                    "default": "instagram",
                    "placeholder": "Folder inside the output directory"
                }),
                "filename": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty for uploader_shortcode"
                }),
                "resolution": (["best", "1080p", "720p", "480p"],),
                "cookies_from_browser": (["none", "chrome", "firefox", "edge", "brave"], {
                    "tooltip": "Instagram blocks most downloads unless you are logged in. Pick the browser you are logged in with."
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "file_path")
    FUNCTION = "download"
    CATEGORY = "denrakeiw/video"

    def download(self, url, subfolder, filename, resolution, cookies_from_browser):
        url = url.strip()
        host = urlparse(url).hostname or ""
        if not (host == "instagram.com" or host.endswith(".instagram.com")):
            raise ValueError(f"Not an Instagram URL: {url}")

        output_dir = os.path.realpath(folder_paths.get_output_directory())
        if os.path.isabs(subfolder) or os.path.splitdrive(subfolder)[0] or subfolder.startswith(("/", "\\")) or ".." in subfolder.replace("\\", "/").split("/"):
            raise ValueError(f"subfolder must be a relative folder inside the output directory: {subfolder}")
        target_dir = os.path.realpath(os.path.join(output_dir, subfolder))
        if os.path.commonpath([output_dir, target_dir]) != output_dir:
            raise ValueError(f"subfolder points outside the output directory: {subfolder}")
        os.makedirs(target_dir, exist_ok=True)

        stem = filename.strip() or "%(uploader_id)s_%(id)s"
        height = {"best": None, "1080p": 1080, "720p": 720, "480p": 480}[resolution]
        options = {
            "outtmpl": os.path.join(target_dir, f"{stem}.%(ext)s"),
            "format": f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b" if height else "bv*+ba/b",
            "merge_output_format": "mp4",
            "restrictfilenames": True,
            "noplaylist": True,
            "windowsfilenames": os.name == "nt",
        }
        if cookies_from_browser != "none":
            options["cookiesfrombrowser"] = (cookies_from_browser,)

        print(f"[InstagramVideoDownloader] Downloading {url}")
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url)

        downloads = info.get("requested_downloads") or []
        if not downloads:
            raise ValueError(f"Instagram returned no downloadable video for {url}")
        file_path = downloads[0]["filepath"]
        print(f"[InstagramVideoDownloader] Saved to: {file_path}")
        return (VideoFromFile(file_path), file_path)
