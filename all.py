import importlib
import os

import sqlitedict
import random
import os

class KPromtGen:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT",),
                "n": ("INT",),
                "replace_underscore": ("BOOLEAN", {"default": False})
            },
        }
    
    @classmethod
    def IS_CHANGED(c, **kwars):
        return True 
        

    RETURN_NAMES = ("prompt",)
    RETURN_TYPES = ("STRING",)

    FUNCTION = "gen"
    CATEGORY = "kandy"

    def __init__(self):
        sqlite_db_path = os.path.join(os.path.dirname(__file__), 'data', 'wd-eva02-large-tagger-v3.sqlite')
        self.z_tokens = sqlitedict.SqliteDict(sqlite_db_path)
        self.rng = random.Random()
        pass

    def gen(self, seed, n, replace_underscore = False):
        """ Add n random elements from zTokens array to the prompt """

        components = []
        zLen = len(self.z_tokens)
        for _ in range(n):
            token = self.z_tokens[self.rng.randint(1, zLen - 1)]['name'] 
            token = token.replace("_", " ") if replace_underscore else token
            components.append(token)
        
        self.rng.shuffle(components)
        prompt = ",".join(components)
        return (prompt,)

        
import re

class KandySimplePrompt: 
    def __init__(self):
        self.rnd = random.Random(324532465345)
        self.ii = 42
        pass
     
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}), 
            },
            "optional":{ 
                "add_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT_TXT",)

    FUNCTION = "node"
    CATEGORY = "KAndy"
 
    def node(self, prompt="", add_prompt="", my_unique_id=0): 
        self.ii += 1
        self.rnd.seed(int(my_unique_id) + self.ii)
        out = self.replace_random_part(prompt  + "," + add_prompt)
        return(out,)

    def replace_random_part(self, text): 
        matches = re.findall(r'\{([^{}]*)\}', text)
         
        for match in matches: 
            options = match.split('|')
            replacement = self.rnd.choice(options) 
            text = text.replace('{' + match + '}', replacement, 1)
         
        text = text.replace('|', '').replace('{', '').replace('}', '')
        
        return text



class KandySave: 
    STORAGE = ''
    
     
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}), 
            },
        }
    
    @classmethod
    def IS_CHANGED(c, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "node"
    CATEGORY = "KAndy"
 
    def node(self, text): 
        KandySave.STORAGE = text
        print(f"[KandySave] Saved text: {text}")
        return (text,)
 
    
class KandyLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_text": ("STRING", {"multiline": True}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "node"
    CATEGORY = "KAndy"
    
    @classmethod
    def IS_CHANGED(c, **kwargs):
        return float("NaN")  # This will always trigger a change, forcing the node to update every time it's used. 
    
    def node(self, default_text): 
        print(f"[KandyLoad] Loaded text: {KandySave.STORAGE}")
        return (KandySave.STORAGE if KandySave.STORAGE else default_text,)




# class AnyType(str):
#     """A special type that can be connected to any other types."""
#     def __ne__(self, __value: object) -> bool:
#         return False
# any_type = AnyType("*")


from .tagger import KAndyWD14Tagger
from .tagger import KAndyTaggerModelLoader

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KPromtGen": KPromtGen,
    "KandySimplePrompt": KandySimplePrompt,
    "KAndyTaggerModelLoader": KAndyTaggerModelLoader,
    "KAndyWD14Tagger": KAndyWD14Tagger,
    "KandySave": KandySave,
    "KandyLoad": KandyLoad,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KPromtGen": "Promt Generator",
    "KandySimplePrompt": "Simple Prompt",
    "KAndyTaggerModelLoader": "Tagger ModelLoader",
    "KAndyWD14Tagger": "WD14 Tagger",
    "KandySave": "Save Text",
    "KandyLoad": "Load Text",
}


nodes_dir = os.path.join(os.path.dirname(__file__), 'nodes')
node_list = [os.path.splitext(f)[0] for f in os.listdir(nodes_dir) if os.path.isfile(os.path.join(nodes_dir, f)) and f.endswith('.py')]

for module in node_list:
    try:
        imported_module = importlib.import_module(f".nodes.{module}", __package__)

        module_key = imported_module.__NODE__.__name__
        module_name = ' '.join(word.capitalize() for word in module.split('-'))
        NODE_CLASS_MAPPINGS[module_key] = imported_module.__NODE__
        NODE_DISPLAY_NAME_MAPPINGS[module_key] = module_name
        print(f"[KAndy] Add node: {module_key} {module}")
    except Exception as e:
        print(f"Failed to import module: {module}" + e)