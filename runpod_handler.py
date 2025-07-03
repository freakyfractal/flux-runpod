"""RunPod Serverless handler for Flux 1‑dev in FP32 with per‑call LoRA fusion/cache."""
import os, hashlib, base64, io, time
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
import runpod

BASE_MODEL = os.getenv("BASE_MODEL", "black-forest-labs/FLUX.1-dev")
CACHE_DIR  = "/data/lora_fused"

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
pipe = AutoPipelineForText2Image.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    token=token  # explicit auth
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# Try enabling xFormers; skip if wheel unavailable
try:
    import xformers  # noqa: F401
    pipe.enable_xformers_memory_efficient_attention()
except (ModuleNotFoundError, RuntimeError):
    print("xformers not available — proceeding without memory‑efficient attention")
os.makedirs(CACHE_DIR, exist_ok=True)


def apply_loras(loras: dict[str, str]):
    """Fuse LoRAs and cache result on local SSD."""
    if not loras:
        pipe.set_adapters([])
        return
    key = hashlib.sha1("".join(sorted(loras.values())).encode()).hexdigest()[:16]
    fused_path = f"{CACHE_DIR}/{key}"
    if os.path.exists(fused_path):
        pipe.load_pretrained(fused_path)
        return
    for name, repo in loras.items():
        pipe.load_lora_weights(repo, adapter_name=name)
    pipe.fuse_lora()
    pipe.save_pretrained(fused_path)
    pipe.unload_lora_weights()


def handler(event):
    inp = event["input"]
    prompt = inp["prompt"]
    neg    = inp.get("negative_prompt", "")
    steps  = int(inp.get("steps", 20))
    cfg    = float(inp.get("cfg", 7.0))
    loras  = inp.get("loras", {})

    apply_loras(loras)

    t0 = time.time()
    img = pipe(prompt,
               negative_prompt=neg,
               num_inference_steps=steps,
               guidance_scale=cfg).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")

    return runpod.serverless.utils.generic_response(200, {
        "latency_sec": round(time.time() - t0, 2),
        "image_png_base64": base64.b64encode(buf.getvalue()).decode()
    })

runpod.serverless.start({"handler": handler})
