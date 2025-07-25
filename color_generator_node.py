import torch
import numpy as np
from PIL import Image


class ColorGeneratorNode:
    """
    A ComfyUI node that generates solid color images based on predefined colors.
    Outputs the color name, hex code, and a solid color image.
    """

    # Predefined color palette with names and hex codes
    COLORS = {
        # Basic Colors
        "Red": "#FF0000",
        "Green": "#008000",
        "Blue": "#0000FF",
        "Yellow": "#FFFF00",
        "Cyan": "#00FFFF",
        "Magenta": "#FF00FF",
        "White": "#FFFFFF",
        "Black": "#000000",

        # Grays
        "Gray": "#808080",
        "Light Gray": "#D3D3D3",
        "Dark Gray": "#A9A9A9",
        "Dim Gray": "#696969",
        "Silver": "#C0C0C0",
        "Gainsboro": "#DCDCDC",
        "WhiteSmoke": "#F5F5F5",

        # Reds
        "Dark Red": "#8B0000",
        "Crimson": "#DC143C",
        "Fire Brick": "#B22222",
        "Indian Red": "#CD5C5C",
        "Light Coral": "#F08080",
        "Salmon": "#FA8072",
        "Dark Salmon": "#E9967A",
        "Light Salmon": "#FFA07A",
        "Coral": "#FF7F50",
        "Tomato": "#FF6347",
        "Orange Red": "#FF4500",
        "Red Orange": "#FF5349",
        "Maroon": "#800000",

        # Oranges
        "Orange": "#FFA500",
        "Dark Orange": "#FF8C00",
        "Peach": "#FFCBA4",
        "Papaya Whip": "#FFEFD5",
        "Moccasin": "#FFE4B5",
        "Peach Puff": "#FFDAB9",
        "Pale Goldenrod": "#EEE8AA",
        "Khaki": "#F0E68C",
        "Dark Khaki": "#BDB76B",

        # Yellows
        "Light Yellow": "#FFFFE0",
        "Lemon Chiffon": "#FFFACD",
        "Light Goldenrod Yellow": "#FAFAD2",
        "Cornsilk": "#FFF8DC",
        "Gold": "#FFD700",
        "Dark Goldenrod": "#B8860B",
        "Goldenrod": "#DAA520",

        # Greens
        "Lime": "#00FF00",
        "Lime Green": "#32CD32",
        "Lawn Green": "#7CFC00",
        "Chartreuse": "#7FFF00",
        "Green Yellow": "#ADFF2F",
        "Spring Green": "#00FF7F",
        "Medium Spring Green": "#00FA9A",
        "Light Green": "#90EE90",
        "Pale Green": "#98FB98",
        "Dark Green": "#006400",
        "Forest Green": "#228B22",
        "Sea Green": "#2E8B57",
        "Medium Sea Green": "#3CB371",
        "Dark Sea Green": "#8FBC8F",
        "Olive": "#808000",
        "Olive Drab": "#6B8E23",
        "Yellow Green": "#9ACD32",

        # Cyans/Teals
        "Aqua": "#00FFFF",
        "Dark Cyan": "#008B8B",
        "Teal": "#008080",
        "Dark Turquoise": "#00CED1",
        "Turquoise": "#40E0D0",
        "Medium Turquoise": "#48D1CC",
        "Pale Turquoise": "#AFEEEE",
        "Aqua Marine": "#7FFFD4",
        "Light Cyan": "#E0FFFF",
        "Cadet Blue": "#5F9EA0",

        # Blues
        "Light Blue": "#ADD8E6",
        "Powder Blue": "#B0E0E6",
        "Light Steel Blue": "#B0C4DE",
        "Steel Blue": "#4682B4",
        "Cornflower Blue": "#6495ED",
        "Deep Sky Blue": "#00BFFF",
        "Dodger Blue": "#1E90FF",
        "Royal Blue": "#4169E1",
        "Medium Blue": "#0000CD",
        "Dark Blue": "#00008B",
        "Navy": "#000080",
        "Midnight Blue": "#191970",
        "Sky Blue": "#87CEEB",
        "Alice Blue": "#F0F8FF",

        # Purples/Violets
        "Purple": "#800080",
        "Dark Purple": "#663399",
        "Indigo": "#4B0082",
        "Dark Slate Blue": "#483D8B",
        "Slate Blue": "#6A5ACD",
        "Medium Slate Blue": "#7B68EE",
        "Medium Purple": "#9370DB",
        "Blue Violet": "#8A2BE2",
        "Dark Violet": "#9400D3",
        "Dark Orchid": "#9932CC",
        "Medium Orchid": "#BA55D3",
        "Orchid": "#DA70D6",
        "Violet": "#EE82EE",
        "Plum": "#DDA0DD",
        "Thistle": "#D8BFD8",
        "Lavender": "#E6E6FA",

        # Pinks
        "Pink": "#FFC0CB",
        "Light Pink": "#FFB6C1",
        "Hot Pink": "#FF69B4",
        "Deep Pink": "#FF1493",
        "Medium Violet Red": "#C71585",
        "Pale Violet Red": "#DB7093",

        # Browns
        "Brown": "#A52A2A",
        "Saddle Brown": "#8B4513",
        "Sienna": "#A0522D",
        "Chocolate": "#D2691E",
        "Dark Goldenrod": "#B8860B",
        "Peru": "#CD853F",
        "Rosy Brown": "#BC8F8F",
        "Sandy Brown": "#F4A460",
        "Tan": "#D2B48C",
        "Burlywood": "#DEB887",
        "Wheat": "#F5DEB3",
        "Navajo White": "#FFDEAD",
        "Bisque": "#FFE4C4",
        "Blanched Almond": "#FFEBCD",
        "Antique White": "#FAEBD7",

        # Special Colors
        "Beige": "#F5F5DC",
        "Floral White": "#FFFAF0",
        "Ghost White": "#F8F8FF",
        "Honeydew": "#F0FFF0",
        "Ivory": "#FFFFF0",
        "Azure": "#F0FFFF",
        "Snow": "#FFFAFA",
        "Mint Cream": "#F5FFFA",
        "Seashell": "#FFF5EE",
        "Old Lace": "#FDF5E6",
        "Linen": "#FAF0E6"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "color": (list(cls.COLORS.keys()), {
                    "default": "Red"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "color_name", "hex_code")
    FUNCTION = "generate_color"
    CATEGORY = "denrakeiw/image"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def generate_color(self, width, height, color):
        """Generate a solid color image with the specified dimensions and color"""

        # Get hex code for the selected color
        hex_code = self.COLORS[color]

        # Convert hex to RGB
        rgb = self.hex_to_rgb(hex_code)

        # Create PIL Image with solid color
        pil_image = Image.new('RGB', (width, height), rgb)

        # Convert PIL image to numpy array
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to torch tensor and add batch dimension
        # ComfyUI expects images in format [batch, height, width, channels]
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        return (image_tensor, color, hex_code)