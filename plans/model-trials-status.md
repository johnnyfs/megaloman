# Model trials status (2026-10-24)

Survey routes #1 and #3 from `docs/pixel-model-survey.md` (#2 FLUX Kontext was the "diffusy" one).
Code: `src/megaloman/modal_pixel_open.py` (Modal app `megaloman-pixel-open`), driver `src/megaloman/open_models_experiment.py`.
Output: `workspace/experiments/regions/snipe/compact/open-models/<model>/pipeline.json`.

| model | snipe 64/32/16 | squid 64/32/16 |
|---|---|---|
| SDXL + Pixel Icon LoRA (L4, img2img from pre-pixelated concept, 4 seeds) | done | done |
| Qwen-Image-Edit-2511 + Render It (H100, concept + NES style sheet refs) | not run | not run |

- SDXL: 64 and 32 read well (clean lattice, cleanliness ~1.0); 16 is mushy blobs (strength 0.6 lost the silhouette).
- Qwen crash loop cause: `ImportError: Qwen2VLVideoProcessor requires the Torchvision library` at model load.
  Fixed in the image (added `torchvision`), not re-run. Weights are already cached in the `megaloman-flux` Volume.
- Next: `open_models_experiment` with `MODELS` limited to qwen-edit (H100, ~12 images); maybe SDXL 16 at lower strength.
- Cost so far ~ $0.6 estimated (SDXL L4 ~$0.15; Qwen H100 crash-looping ~10 min ~$0.4-0.6); timings lost when the driver was stopped.
