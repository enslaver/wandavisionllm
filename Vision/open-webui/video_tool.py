"""
title: Video Generation (ComfyUI)
author: wandavision
version: 0.1.0
description: Text -> video with sound. MiniMax H3 + 8-step turbo LoRA on your ComfyUI (the GPU box); the MP4 is attached to the chat message.
"""

import asyncio
import io
import json
import random
import time
import uuid

import aiohttp
from fastapi import UploadFile
from pydantic import BaseModel, Field

from open_webui.models.chats import Chats
from open_webui.models.users import UserModel
from open_webui.routers.files import upload_file_handler
from open_webui.utils.chat_id import is_saved_chat_id

# API-format copy of Vision/comfyui/workflows/minimax-h3-t2v.api.json (ComfyUI template video_minimax_h3_t2v, turbo on)
WORKFLOW = {
    "127": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
    "134": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["127", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1}},
    "128": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
    "119": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
    "120": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    "115": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4, "multiple": 32}},
    "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["128", 0], "vae": ["119", 0], "prompt": "", "width": ["115", 0], "height": ["115", 1], "length": 124}},
    "126": {"class_type": "BasicGuider", "inputs": {"model": ["134", 0], "conditioning": ["131", 0]}},
    "124": {"class_type": "BasicScheduler", "inputs": {"model": ["134", 0], "scheduler": "simple", "steps": 8, "denoise": 1}},
    "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
    "125": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["129", 0], "guider": ["126", 0], "sampler": ["123", 0], "sigmas": ["124", 0], "latent_image": ["131", 1]}},
    "122": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
    "121": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
    "130": {"class_type": "CreateVideo", "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": 24}},
    "92": {"class_type": "SaveVideo", "inputs": {"video": ["130", 0], "filename_prefix": "video/open-webui", "format": "auto", "format.codec": "auto"}},
}
ASPECTS = ["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)",
           "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)"]


def frames(seconds: float) -> int:
    """24 fps, snapped up to MiniMax H3's 17k+5 frame grid (5 s -> 124)."""
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


class Tools:
    class Valves(BaseModel):
        COMFYUI_URL: str = Field("http://gpu-box.example.ts.net", description="ComfyUI base URL: the GPU box's Caddy, e.g. http://gpu-box.local on the LAN")
        DEFAULT_SECONDS: float = Field(5, description="Clip length when the model doesn't pick one")
        MAX_SECONDS: float = Field(10, description="Longest clip allowed")
        ASPECT_RATIO: str = Field("16:9 (Widescreen)", description="One of: " + ", ".join(ASPECTS))
        MEGAPIXELS: float = Field(0.4, description="0.4 -> 864x480 at 16:9; higher is much slower on 12 GB")
        TIMEOUT_S: int = Field(1800, description="Give up after this many seconds")

    def __init__(self):
        self.valves = self.Valves()

    async def generate_video(
        self,
        prompt: str,
        seconds: float = 0,
        __request__=None,
        __user__: dict = None,
        __event_emitter__=None,
        __chat_id__: str = None,
        __message_id__: str = None,
    ) -> str:
        """
        Generate a short video clip with sound from a text prompt. Takes 1-5 minutes.

        :param prompt: Detailed description of the clip: subject, action and motion, camera, lighting, style, and the sounds or music to hear.
        :param seconds: Clip length in seconds (2-10). Use 0 for the default of 5.
        :return: Whether the video was generated; the video is attached to the chat for the user.
        """
        v = self.valves
        base = v.COMFYUI_URL.rstrip("/")
        secs = min(max(seconds or v.DEFAULT_SECONDS, 1), v.MAX_SECONDS)

        async def status(text, done=False):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": text, "done": done}})

        wf = json.loads(json.dumps(WORKFLOW))
        wf["131"]["inputs"]["prompt"] = prompt
        wf["131"]["inputs"]["length"] = frames(secs)
        wf["115"]["inputs"]["aspect_ratio"] = v.ASPECT_RATIO if v.ASPECT_RATIO in ASPECTS else "16:9 (Widescreen)"
        wf["115"]["inputs"]["megapixels"] = v.MEGAPIXELS
        wf["129"]["inputs"]["noise_seed"] = random.randint(0, 2**48)

        start = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(base + "/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())}) as r:
                    body = await r.json(content_type=None)
                    if r.status != 200 or "prompt_id" not in body:
                        raise RuntimeError("ComfyUI rejected the workflow: %s" % json.dumps(body)[:500])
                pid = body["prompt_id"]
                await status("Rendering a %gs video in ComfyUI..." % secs)

                outputs = None
                while time.time() - start < v.TIMEOUT_S:
                    await asyncio.sleep(3)
                    async with s.get(base + "/history/" + pid) as r:
                        hist = await r.json(content_type=None)
                    if pid in hist:
                        st = hist[pid].get("status", {})
                        if st.get("status_str") != "success":
                            err = [m[1] for m in st.get("messages", []) if m[0] == "execution_error"]
                            raise RuntimeError("ComfyUI error: %s" % (err[0].get("exception_message", "") if err else st))
                        outputs = hist[pid]["outputs"]
                        break
                    await status("Rendering a %gs video in ComfyUI... %ds" % (secs, time.time() - start))
                if outputs is None:
                    raise RuntimeError("timed out after %ds (job %s is still in ComfyUI's queue)" % (v.TIMEOUT_S, pid))

                item = next(i for o in outputs.values() for i in o.get("images", []) if i["filename"].endswith((".mp4", ".webm", ".mkv")))
                params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                async with s.get(base + "/view", params=params, timeout=aiohttp.ClientTimeout(total=300)) as r:
                    r.raise_for_status()
                    data = await r.read()
                    ctype = r.headers.get("content-type", "video/mp4").split(";")[0]

            user = UserModel(**__user__) if __user__ else None
            ext = "." + item["filename"].rsplit(".", 1)[-1]
            meta = {"chat_id": __chat_id__, "message_id": __message_id__, "prompt": prompt}
            file_item = await upload_file_handler(
                __request__,
                file=UploadFile(file=io.BytesIO(data), filename="generated-video" + ext, headers={"content-type": ctype}),
                metadata=meta,
                process=False,
                user=user,
            )
            url = __request__.app.url_path_for("get_file_content_by_id", id=file_item.id)
            files = [{"type": "video", "id": file_item.id, "url": url, "name": file_item.filename, "content_type": ctype}]
            if is_saved_chat_id(__chat_id__) and __message_id__:
                saved = await Chats.add_message_files_by_id_and_message_id(__chat_id__, __message_id__, files)
                files = saved or files
            if __event_emitter__:
                await __event_emitter__({"type": "chat:message:files", "data": {"files": files}})
            took = time.time() - start
            await status("Video ready in %ds" % took, done=True)
            return json.dumps({
                "status": "success",
                "message": "The video was generated (%gs, %d KB, %ds) and is already attached to the chat. Do not embed it again; just tell the user it is ready." % (secs, len(data) // 1024, took),
                "url": url,
            })
        except Exception as e:
            await status("Video failed: %s" % e, done=True)
            return json.dumps({"error": str(e)})
