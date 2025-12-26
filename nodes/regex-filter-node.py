import re

class RegexFilterNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
                "multiline_string": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_strings",)
    FUNCTION = "filter_strings"
    CATEGORY = "kandy"

    def filter_strings(self, data, multiline_string):
        output_strings = []
        lines = multiline_string.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split('@', 2) # Split by '@' at most twice

            if len(parts) == 3:
                regex1_pattern = parts[0]
                regex2_pattern = parts[1]
                target_string = parts[2]

                try:
                    match1 = re.search(regex1_pattern, data)
                    match2 = re.search(regex2_pattern, data)

                    if match1 and not match2:
                        return (target_string,)
                except re.error as e:
                    print(f"Regex error in line '{line}': {e}")

            else:
                print(f"Skipping malformed line: '{line}'")

        return ("\n".join(output_strings),)

__NODE__ = RegexFilterNode
