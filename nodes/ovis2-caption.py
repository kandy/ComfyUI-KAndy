import os
import torch
import numpy as np
import folder_paths
from PIL import Image
from transformers import AutoModelForCausalLM

# Register model_path
models_dir = os.path.join(folder_paths.base_path, "models")
ovis_dir = os.path.join(models_dir, "ovis")
os.makedirs(ovis_dir, exist_ok=True)

# Add Ovis models folder to folder_paths
if "ovis" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ovis"] = ([ovis_dir], folder_paths.supported_pt_extensions)


class Ovis2ImageCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "KAndy"

    def generate_caption(self, model, image, prompt, max_new_tokens, temperature, thinking=False):
        # Convert ComfyUI image to PIL image format
        # ComfyUI images are in format [batch, height, width, channel]
        if len(image.shape) == 4:
            # Handle batch of images - take the first one
            image_tensor = image[0]
        else:
            # Handle single image
            image_tensor = image
            
        pil_image = Image.fromarray((image_tensor * 255).cpu().numpy().astype(np.uint8))
        

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]
        
        # Thinking mode & budget
        enable_thinking = thinking
        enable_thinking_budget = thinking  # Only effective if enable_thinking is True.

        # Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
        thinking_budget = max_new_tokens // 2 if thinking else 0

        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        device = model.device
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)if pixel_values is not None else None
        grid_thws = grid_thws.to(device) if grid_thws is not None else None

        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
        )

        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(response)
            
        return (response,)
    
__NODE__ = Ovis2ImageCaption