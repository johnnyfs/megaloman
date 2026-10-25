"""The two remaining open routes from docs/pixel-model-survey.md ("Top 3 to try on Modal", #1 and #3) on Modal.

    uv run --with modal modal deploy src/megaloman/modal_pixel_open.py
    (driver: src/megaloman/open_models_experiment.py)

- SdxlIcon: SDXL base + Pixel Icon LoRA (lizozom, trigger `yotoicon`, 16 cells per 512 px), img2img from a
  pre-pixelated concept. L4.
- QwenEdit: Qwen-Image-Edit-2511 + Render It LoRA (AmirKerr, "Render this pixelart, ..."), multi-reference
  edit (pre-pixelated concept + style exemplar sheet). H100 (bf16 ~58 GB).
Weights cache in the shared `megaloman-flux` Volume (HF cache).
"""
import io
import time

import modal

MODEL_DIR = "/models"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
ICON_LORA = "lizozom/pixel-icon-lora-sdxl"
QWEN = "Qwen/Qwen-Image-Edit-2511"
RENDER_LORA, RENDER_FILE = "AmirKerr/Render_It", "Qwen_Edit_Image_Render_It_Alpha.safetensors"
USD_PER_HOUR = {"L4": 0.80, "H100": 3.95}

volume = modal.Volume.from_name("megaloman-flux", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.10.0", "torchvision", "diffusers==0.40.0", "transformers>=4.57", "accelerate", "sentencepiece", "protobuf",
    "peft", "pillow", "huggingface_hub[hf_transfer]",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{MODEL_DIR}/hf"})
app = modal.App("megaloman-pixel-open")


def _png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@app.function(image=image, volumes={MODEL_DIR: volume}, timeout=3600)
def download():
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(SDXL, allow_patterns=["*.json", "*.txt", "*fp16.safetensors"], ignore_patterns=["*refiner*"])
    snapshot_download(SDXL_VAE)
    snapshot_download(ICON_LORA)
    snapshot_download(QWEN)
    hf_hub_download(RENDER_LORA, RENDER_FILE)
    volume.commit()


@app.cls(image=image, gpu="L4", volumes={MODEL_DIR: volume}, timeout=1800, scaledown_window=60)
class SdxlIcon:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
        t = time.time()
        vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.load_lora_weights(ICON_LORA, adapter_name="icon")
        self.load_s = time.time() - t

    @modal.method()
    def img2img(self, png, prompt, negative, seeds, size, strength, lora_scale=0.7, steps=30, guidance=7.0):
        import torch
        from PIL import Image
        self.pipe.set_adapters(["icon"], adapter_weights=[lora_scale])
        src = Image.open(io.BytesIO(png)).convert("RGB").resize((size, size), Image.NEAREST)
        out, t = [], time.time()
        for s in seeds:
            im = self.pipe(image=src, prompt=prompt, negative_prompt=negative, strength=strength,
                           num_inference_steps=steps, guidance_scale=guidance,
                           generator=torch.Generator("cuda").manual_seed(s)).images[0]
            out.append(_png(im))
        return {"images": out, "gen_s": time.time() - t, "load_s": self.load_s}


@app.cls(image=image, gpu="H100", volumes={MODEL_DIR: volume}, timeout=3600, scaledown_window=60)
class QwenEdit:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        t = time.time()
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.load_lora_weights(RENDER_LORA, weight_name=RENDER_FILE, adapter_name="render")
        self.load_s = time.time() - t

    @modal.method()
    def edit(self, pngs, prompt, seeds, size=768, lora_scale=1.0, steps=25, cfg=4.0):
        import torch
        from PIL import Image
        self.pipe.set_adapters(["render"], adapter_weights=[lora_scale])
        ims = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
        out, t = [], time.time()
        for s in seeds:
            im = self.pipe(image=ims, prompt=prompt, negative_prompt=" ", true_cfg_scale=cfg,
                           num_inference_steps=steps, width=size, height=size,
                           generator=torch.Generator("cuda").manual_seed(s)).images[0]
            out.append(_png(im))
        return {"images": out, "gen_s": time.time() - t, "load_s": self.load_s}
