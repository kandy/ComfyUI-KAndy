import comfy.utils
import numpy as np
import csv
import os
from PIL import Image
import onnxruntime as ort

from folder_paths import models_dir
import time

models = {
    "wd-eva02-large-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnext-tagger": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-vit-tagger",
    "Z3D-E621-Convnext": "{HF_ENDPOINT}/silveroxides/Z3D-E621-Convnext",
}


class KAndyTaggerModelLoader:
    """
    Load a model for KAndy Tagger.
    This node is used to load a model for KAndy Tagger.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(models.keys()),),
                "target": (["CUDAExecutionProvider", "CPUExecutionProvider"],),
            }
        }

    RETURN_TYPES = ("KModelLoader",)

    FUNCTION = "load_model"
    CATEGORY = "KAndy"

    def load_model(self, model, target="CUDAExecutionProvider"):
        global models_dir
        wd14_dir = os.path.join(models_dir, "wd14")
        print(f"Model path: {wd14_dir}")
        name = os.path.join(wd14_dir, model + ".onnx")
        if not os.path.exists(name):
            raise FileNotFoundError(f"Model file {name} does not exist.")

        model_session = ort.InferenceSession(name, providers=[target])
        return ((model_session, os.path.join(wd14_dir, model + ".csv")),)


def _tag(
    image,
    models,
    threshold=0.35,
    character_threshold=0.85,
    exclude_tags="",
    replace_underscore=True,
    trailing_comma=False,
    client_id=None,
    node=None,
):
    (model, path) = models

    input = model.get_inputs()[0]
    height = input.shape[1]

    # Reduce to max size and pad with white
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    # Read all tags from csv and locate start of each category
    tags = []
    general_index = None
    character_index = None
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [
        item for item in result[general_index:character_index] if item[1] > threshold
    ]
    character = [
        item for item in result[character_index:] if item[1] > character_threshold
    ]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ("" if trailing_comma else ", ").join(
        (
            item[0].replace("(", "\\(").replace(")", "\\)")
            + (", " if trailing_comma else "")
            for item in all
        )
    )

    print(res)
    return res


class KAndyWD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        defaults = {
            "model": "wd-v1-4-moat-tagger-v2",
            "threshold": 0.35,
            "character_threshold": 0.85,
            "replace_underscore": False,
            "trailing_comma": False,
            "exclude_tags": "",
            "ortProviders": ["CUDAExecutionProvider"],
            "HF_ENDPOINT": "https://huggingface.co",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("KModelLoader",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": defaults["threshold"],
                        "min": 0.0,
                        "max": 1,
                        "step": 0.05,
                    },
                ),
                "character_threshold": (
                    "FLOAT",
                    {
                        "default": defaults["character_threshold"],
                        "min": 0.0,
                        "max": 1,
                        "step": 0.05,
                    },
                ),
                "replace_underscore": (
                    "BOOLEAN",
                    {"default": defaults["replace_underscore"]},
                ),
                "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
                "cache": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"

    OUTPUT_NODE = True

    CATEGORY = "KAndy"

    def tag(
        self,
        image,
        model,
        threshold,
        character_threshold,
        exclude_tags="",
        replace_underscore=False,
        trailing_comma=False,
        cache=False,
    ):
        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(
                _tag(
                    image,
                    model,
                    threshold,
                    character_threshold,
                    exclude_tags,
                    replace_underscore,
                    trailing_comma,
                )
            )
            pbar.update(1)
        return (tags,)
