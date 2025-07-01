"""
RunPod Serverless handler for Flux 1-dev in FP32 with per-call LoRA fusion/cache.
"""
import os, json, hashlib, shutil, tempfile, base64, io, time
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
import runpod

BASE_MODEL = os.getenv("BASE_MODEL", "black-forest-labs/FLUX.1-dev")
CACHE_DIR  = "/data/lora_fused"

pipe = AutoPipelineForText2Image.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    variant=None  # full precision
).to("cuda")

# Switch to LCMScheduler for faster inference
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

os.makedirs(CACHE_DIR, exist_ok=True)
pipe.enable_xformers_memory_efficient_attention()  # works on FP32

def apply_loras(loras: dict[str, str]):
    """
    Fuse one or more LoRAs and cache the fused pipeline on local SSD.
    `loras` maps arbitrary names to public HF LoRA repos.
    """
    if not loras:
        pipe.set_adapters([])  # ensure base weights
        return

    # Deterministic cache key
    key = hashlib.sha1("".join(sorted(loras.values())).encode()).hexdigest()[:16]
    fused_path = f"{CACHE_DIR}/{key}"

    if os.path.exists(fused_path):
        pipe.load_pretrained(fused_path)
        return

    # Fresh fusion
    for name, repo in loras.items():
        pipe.load_lora_weights(repo, adapter_name=name)
    pipe.fuse_lora()
    pipe.save_pretrained(fused_path)
    # Clean adapters from RAM
    pipe.unload_lora_weights()

def handler(event):
    """
    Expected JSON body:
    {
      "prompt": "...",
      "negative_prompt": "...",          # optional
      "loras": { "style": "org/repo" },  # optional
      "steps": 20,                       # optional (default 20)
      "cfg": 7.0                         # optional (default 7.0)
    }
    Returns base-64 PNG.
    """
    inp = event["input"]
    prompt  = inp["prompt"]
    neg     = inp.get("negative_prompt", "")
    steps   = int(inp.get("steps", 20))
    cfg     = float(inp.get("cfg", 7.0))
    loras   = inp.get("loras", {})

    apply_loras(loras)

    t0 = time.time()
    out = pipe(
        prompt,
        negative_prompt=neg,
        num_inference_steps=steps,
        guidance_scale=cfg
    )
    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return runpod.serverless.utils.generic_response(
        200,
        {
            "latency_sec": round(time.time() - t0, 2),
            "image_png_base64": base64.b64encode(buf.getvalue()).decode()
        }
    )

runpod.serverless.start({"handler": handler})

