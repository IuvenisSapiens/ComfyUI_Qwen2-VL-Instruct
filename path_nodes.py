from decord import VideoReader, cpu  # pip install decord
class MultiplePathsInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "path_1": ("PATH",),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine"
    CATEGORY = "Comfyui_Qwen2-VL-Instruct"
    DESCRIPTION = """
Creates a path batch from multiple paths.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""



    @staticmethod
    def convert_path_to_json(file_path):
        ext = file_path.split('.')[-1].lower()
        
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
            return {"type": "image", "image": f"{file_path}"}
        elif ext in ["mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"]:
            print("source_video_path:", file_path)
            vr = VideoReader(file_path, ctx=cpu(0))
            total_frames = len(vr) + 1
            print("Total frames:", total_frames)
            avg_fps = vr.get_avg_fps()
            print("Get average FPS(frame per second):", avg_fps)
            duration = len(vr) / avg_fps
            print("Total duration:", duration, "seconds")
            width = vr[0].shape[1]
            height = vr[0].shape[0]
            print("Video resolution(width x height):", width, "x", height)
            return {
                "type": "video",
                "video": f"{file_path}",
                "fps": 1.0,
            }
        else:
            return None



    def combine(self, inputcount, **kwargs):
        path_list = []
        for c in range(inputcount):
            path = kwargs[f"path_{c + 1}"]
            path = self.convert_path_to_json(path)
            print(path)
            path_list.append(path)
        print(path_list)
        result = path_list
        return (result,)