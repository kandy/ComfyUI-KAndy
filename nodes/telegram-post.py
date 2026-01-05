import requests
import numpy as np
from PIL import Image
import io


class TelegramBotPost:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bot_token": ("STRING", {"multiline": False}),
                "chat_id": ("STRING", {"multiline": False}),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "post_image"
    OUTPUT_NODE = True
    CATEGORY = "KAndy"

    def post_image(self, image, bot_token, chat_id, text):
        if not bot_token or not chat_id:
            print("Telegram Bot: bot_token or chat_id is empty. Skipping.")
            return ()

        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto?chat_id={chat_id}&caption={text}"

        # Convert tensor to PIL Image
        i = 255.0 * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Save image to a byte buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        files = {"photo": buffer}

        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Telegram Bot: Image posted successfully!")
                print(response.json())
            else:
                print(f"Telegram Bot: Error posting image: {response.text}")
                print(f"Telegram Bot: Status Code: {response.status_code}")
                print("Please check your bot_token and chat_id.")
        except requests.exceptions.RequestException as e:
            print(f"Telegram Bot: Error posting image: {e}")

        return ()


__NODE__ = TelegramBotPost
