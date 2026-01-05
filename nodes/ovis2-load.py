import os
import torch
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM

# Register model_path
models_dir = os.path.join(folder_paths.base_path, "models")
ovis_dir = os.path.join(models_dir, "ovis")
os.makedirs(ovis_dir, exist_ok=True)

# Add Ovis models folder to folder_paths
if "ovis" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ovis"] = (
        [ovis_dir],
        folder_paths.supported_pt_extensions,
    )


class Ovis2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["AIDC-AI/Ovis2.5-2B"],
                    {"default": "AIDC-AI/Ovis2.5-2B"},
                ),
                "precision": (
                    ["bfloat16", "float16", "float32"],
                    {"default": "bfloat16"},
                ),
                "max_token_length": (
                    "INT",
                    {"default": 32768, "min": 2048, "max": 65536},
                ),
                "device": (
                    [
                        "cuda",
                        "cpu",
                        "cuda:0",
                        "cuda:1",
                    ],
                    {"default": "cuda"},
                ),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("OVIS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "KAndy"

    def download_model(self, model_name):
        """Download the model files from Hugging Face if they don't exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split("/")[-1])

        print(f"Downloading Ovis2 model: {model_name} to {local_dir}")
        try:
            # Create a complete snapshot of the repository locally
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # Use actual files instead of symlinks for better compatibility
            )
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(
                f"Failed to download model {model_name}. Error: {str(e)}"
            )

    def check_model_files(self, model_name):
        """Check if the model files already exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split("/")[-1])

        # Check for config.json as a basic indicator that the model exists
        config_path = os.path.join(local_dir, "config.json")
        return os.path.exists(config_path), local_dir

    def load_model(
        self, model_name, precision, max_token_length, device, auto_download
    ):
        print(f"Loading Ovis2 model: {model_name}")

        # Set precision
        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Check if model exists locally
        model_exists, local_dir = self.check_model_files(model_name)

        # Download model if it doesn't exist and auto_download is enabled
        if not model_exists and auto_download == "enable":
            self.download_model(model_name)

        # Load the model
        try:
            # First try loading from local directory
            if model_exists or auto_download == "enable":
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        local_dir,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True,
                    ).to(device)
                except Exception as e:
                    print(
                        f"Error loading from local directory, falling back to HuggingFace: {str(e)}"
                    )
                    # Fall back to loading directly from HuggingFace
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True,
                    ).to(device)
            else:
                # Load directly from HuggingFace
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    multimodal_max_length=max_token_length,
                    trust_remote_code=True,
                ).to(device)

            # Get tokenizers
            # text_tokenizer = model.get_text_tokenizer()
            # visual_tokenizer = model.get_visual_tokenizer()

            return (model,)
        except Exception as e:
            print(f"Error loading Ovis2 model: {str(e)}")
            raise e


__NODE__ = Ovis2ModelLoader
