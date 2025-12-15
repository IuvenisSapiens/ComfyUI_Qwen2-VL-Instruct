import os
import torch
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import comfy.model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path
import json

NODE_DIR = Path(__file__).resolve().parent
EXTERNAL_MODELS_PATH = NODE_DIR / "external_models.json"

def _load_external_models() -> dict:
    if not EXTERNAL_MODELS_PATH.exists():
        return {}
    try:
        data = json.loads(EXTERNAL_MODELS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[ComfyUI_Qwen3-VL-Instruct] Could not read external_models.json: {e}")
        return {}
    
EXTERNAL_MODELS = _load_external_models()

def _model_choices():
    built_in = [
        "Qwen3-VL-4B-Instruct-FP8",
        "Qwen3-VL-4B-Thinking-FP8",
        "Qwen3-VL-8B-Instruct-FP8",
        "Qwen3-VL-8B-Thinking-FP8",
        "Qwen3-VL-4B-Instruct",
        "Qwen3-VL-4B-Thinking",
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-8B-Thinking",
    ]
    for k in EXTERNAL_MODELS.keys():
        if k not in built_in:
            built_in.append(k)
    return built_in

DEFAULT_HF_ORG = "Qwen"

def _repo_id_from_selection(selection: str) -> str:
    if "/" in selection:
        return selection
    spec = EXTERNAL_MODELS.get(selection)
    if isinstance(spec, dict):
        repo_id = spec.get("repo_id")
        if isinstance(repo_id, str) and repo_id.strip():
            return repo_id.strip()
    return f"{DEFAULT_HF_ORG}/{selection}"

def _ignore_patterns_for_selection(selection: str) -> list[str] | None:
    spec = EXTERNAL_MODELS.get(selection)
    if isinstance(spec, dict):
        pats = spec.get("ignore_patterns")
        if isinstance(pats, list) and pats:
            return pats
    return None

def _local_folder_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")

def _is_model_dir_complete(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    if not os.path.isfile(os.path.join(path, "config.json")):
        return False

    st_index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(st_index):
        try:
            with open(st_index, "r", encoding="utf-8") as f:
                data = json.load(f)
            shard_files = set(data.get("weight_map", {}).values())
            if not shard_files:
                return False
            return all(os.path.isfile(os.path.join(path, sf)) for sf in shard_files)
        except Exception:
            return False

    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return True
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return True

    return False

class Qwen3_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.current_model_id = None  # Track the current model id
        self.current_quantization = None  # Track the current quantization

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (_model_choices(), {"default": "Qwen3-VL-4B-Instruct-FP8"}),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                ),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def inference(
        self,
        text,
        model,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
        seed,
        quantization,
        source_path=None,
        image=None,  # add image parameter
        attention="eager",
    ):
        if seed != -1:
            torch.manual_seed(seed)
        model_id = _repo_id_from_selection(model)
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", _local_folder_name(model_id)
        )

        if not _is_model_dir_complete(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            ignore_patterns = _ignore_patterns_for_selection(model)

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
                ignore_patterns=ignore_patterns,
            )

        # If model_id or quantization changed, reload processor and model
        if (
            self.current_model_id != model_id
            or self.current_quantization != quantization
            or self.processor is None
            or self.model is None
        ):
            self.current_model_id = model_id
            self.current_quantization = quantization
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

        temp_path = None
        if image is not None:
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
            pil_image.save(temp_path)

        with torch.no_grad():
            if source_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            # Inference: Generation of the output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.processor  # release processor memory
                del self.model  # release model memory
                self.processor = None  # set processor to None
                self.model = None  # set model to None
                self.current_model_id = None
                self.current_quantization = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release GPU memory
                    torch.cuda.ipc_collect()

            return (result,)
