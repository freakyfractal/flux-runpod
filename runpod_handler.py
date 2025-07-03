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
        # No LoRAs requested — keep base weights untouched
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
    """Main RunPod handler."""
    # RunPod wraps payloads as {"input": {...}}; fall back to raw JSON if wrapper missing.
    inp = event.get("input", event)

            prompt = inp["prompt"]
    neg    = inp.get("negative_prompt")  # optional
    steps   = int(inp.get("steps", 20))
    cfg     = float(inp.get("cfg", 7.0))
    seed    = inp.get("seed")            # optional
    loras   = inp.get("loras", {})

    apply_loras(loras)

    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
    }
    sig = pipe.__call__.__code__.co_varnames  # pipeline signature

    # Optional params
    if neg is not None and "negative_prompt" in sig:
        call_kwargs["negative_prompt"] = neg
    if "guidance_scale" in sig:
        call_kwargs["guidance_scale"] = cfg
    if seed is not None and "generator" in sig:
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        call_kwargs["generator"] = gen
    if neg is not None and "negative_prompt" in sig:
        call_kwargs["negative_prompt"] = neg
    if "guidance_scale" in sig:
        call_kwargs["guidance_scale"] = cfg

    t0 = time.time()
    img = pipe(**call_kwargs).images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return runpod.serverless.utils.generic_response(
        200,
        {
            "latency_sec": round(time.time() - t0, 2),
            "image_png_base64": base64.b64encode(buf.getvalue()).decode()
        }
    )

runpod.serverless.start({"handler": handler})({"handler": handler})
