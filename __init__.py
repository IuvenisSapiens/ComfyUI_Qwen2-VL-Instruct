from .nodes import Qwen2_VQA
from .util_nodes import ImageLoader
from .path_nodes import MultiplePathsInput
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen2_VQA": Qwen2_VQA,
    "ImageLoader": ImageLoader,
    "MultiplePathsInput": MultiplePathsInput,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_VQA": "Qwen2 VQA",
    "ImageLoader": "Load Image Advanced",
    "MultiplePathsInput": "Multiple Paths Input",
}