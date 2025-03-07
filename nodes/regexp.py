import re
from typing import Any, Dict

class RegexpMatchNode:
    """"
    "" A node to match a regular expression against a given text.
    "" It returns whether a match is found and the matched text based on the specified group ID.
    "" The node uses Python's re module for regex operations.
    """

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "regexp": ("STRING", {"default": ""}),
                "group_id": ("INT", {"default": 0, "min": 0}),
            }
        }
  
    RETURN_TYPES = ("BOOL", "STRING")
    RETURN_NAMES = ("match", "matched_text")

    FUNCTION = "fmatch"
    CATEGORY = "kandy"

    def fmatch(text: str, regexp: str, group_id: int) -> Dict[str, Any]:
        try:
            pattern = re.compile(regexp)
            match = pattern.search(text)
            if match:
                matched_text = match.group(group_id) if group_id < len(match.groups()) + 1 else ""
                return (True, matched_text)
            else:
                return (False, "")
        except re.error as e:
            return (False, f"Error: {str(e)}")

__NODE__ = RegexpMatchNode
