from PIL import Image
import json


class Workflow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING",),
                "node_id": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("worflow",)
    FUNCTION = "run"
    CATEGORY = "Kandy"

    def run(self, image, node_id="44"):
        image_path = image
        w = ""
        node_id = int(node_id)
        try:
            with Image.open(image_path) as img:
                # extract worflow writed by compfyui in png
                w = str(img.info)
                # Find the start and end indices of the JSON data
                start_index = w.find('"nodes": ') + len('"nodes": ')
                end_index = w.rfind(', "links": ')  # Include the closing brace
                w = w[start_index:end_index]
                w = w.replace("\\'", "'").replace("\\\\", "\\").replace("\\u00e9", "")
                # Parse the JSON string into a Python dictionary
                workflow_data = json.loads(w)
                node = next((i for i in workflow_data if i.get("id") == node_id), None)

                prompt = "".join(node.get("widgets_values")[0])

                if prompt:
                    print("Extracted prompt:", prompt)
                    w = str(prompt)
                else:
                    print("No 'prompt' field found in the JSON data.")

                # workflow_data = json.loads(w)]
        except AttributeError as e:
            print(f"Error: {e} {w}")
            w = ""
        except Exception as e:
            print(f"Error: {e} {w}")
            w = ""

        return (w,)


__NODE__ = Workflow
