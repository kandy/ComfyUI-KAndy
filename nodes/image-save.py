import folder_paths as comfy_paths
import time
import comfy
import re
import os

import piexif
import piexif.helper
from PIL import Image
import numpy as np

# Originally From ComfyUI/nodes.py

class TextTokens:
    def __init__(self):

        self.custom_tokens = {}

        self.tokens = {
            '[time]': str(time.time()).replace('.','_'),
            '[cuda_device]': str(comfy.model_management.get_torch_device()),
            '[cuda_name]': str(comfy.model_management.get_torch_device_name(device=comfy.model_management.get_torch_device())),
        }

        if '.' in self.tokens['[time]']:
            self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

        try:
            self.tokens['[user]'] = os.getlogin() if os.getlogin() else 'null'
        except Exception:
            self.tokens['[user]'] = 'null'

    def addToken(self, name, value):
        self.custom_tokens.update({name: value})
        self._update()

    def removeToken(self, name):
        self.custom_tokens.pop(name)
        self._update()

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()
        if self.custom_tokens:
            tokens.update(self.custom_tokens)

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            pattern = re.compile(re.escape(token))
            text = pattern.sub(value, text)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)

        return text

    def _update(self):
        pass


class KAndyImageSave:
    def __init__(self):
        self.output_dir = comfy_paths.output_directory
        self.type = 'output'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "positive": ("STRING",  {"default": '', "multiline": True, "tooltip": "positive prompt"}),
                "negative": ("STRING",  {"default": '', "multiline": True, "tooltip": "negative prompt"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "files",)

    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "KAndy"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    def save_images(self, images, output_path='', filename_prefix="ComfyUI", filename_delimiter='_',
                        positive = '', negative='',
                        extension='jpg', dpi=300, quality=100, optimize_image="true", lossless_webp="false", prompt=None, extra_pnginfo=None,
                        overwrite_mode='false', filename_number_padding=4, filename_number_start='false',
                        show_previews="false"):

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        file_extension = '.' + extension

        # Define token system
        tokens = TextTokens()

        original_output = self.output_dir
        # Parse prefix tokens
        filename_prefix = tokens.parseTokens(filename_prefix)

        # Setup output path
        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)

        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)



        if not os.path.exists(output_path.strip()):
            print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path.strip(), exist_ok=True)

        # Find existing counter values
        if filename_number_start == 'true':
            pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"


        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(output_path)
            if re.match(pattern, os.path.basename(filename))
        ]
        existing_counters.sort(reverse=True)

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        results = list()
        output_files = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate the filename stuffs
            if overwrite_mode == 'prefix_as_filename':
                file = f"{filename_prefix}{file_extension}"
            else:
                if filename_number_start == 'true':
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                if os.path.exists(os.path.join(output_path, file)):
                    counter += 1

        
            output_file = os.path.abspath(os.path.join(output_path, file))
            img.save(output_file, quality=quality, optimize=(optimize_image == "true"), dpi=(dpi, dpi))
            

            a111_params = positive + "\n" + negative

            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(a111_params, encoding="unicode")
                },
            })
            piexif.insert(exif_bytes, output_file)

            output_files.append(output_file)

        
            if overwrite_mode == 'false':
                counter += 1

        if show_previews == 'true':
            return {"ui": {"images": results, "files": output_files}, "result": (images, output_files,)}
        else:
            return {"ui": {"images": []}, "result": (images, output_files,)}


__NODE__ = KAndyImageSave