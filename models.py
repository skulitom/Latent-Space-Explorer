import os
import random
import torch
from diffusers import FluxPipeline

class FluxModel:
    def __init__(self, model_cfg):
        self.device = torch.device(f"cuda:{model_cfg.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
        self.flux_pipeline = self._load_flux_pipeline(model_cfg["model_id"])
        self.diffusion_steps = model_cfg.get("diffusion_steps", 4)
        self.guidance_scale = model_cfg.get("guidance_scale", 0.)
        self.height = model_cfg.get("height", 1024)
        self.width = model_cfg.get("width", 1024)
        self.max_sequence_length = model_cfg.get("max_sequence_length", 256)

    def _load_flux_pipeline(self, model_id):
        print("Loading Flux model...")
        flux_pipeline = FluxPipeline.from_pretrained(model_id, revision='refs/pr/1', torch_dtype=torch.float16)
        flux_pipeline.enable_sequential_cpu_offload()
        flux_pipeline.vae.enable_slicing()
        flux_pipeline.vae.enable_tiling()
        print("Flux model loaded\n")
        return flux_pipeline

    @torch.no_grad()
    def encode_prompt(self, prompt):
        # Flux doesn't have a separate encoding step, so we'll return the prompt as is
        return prompt

    @torch.no_grad()
    def run_flux_inference(self, prompt, latent_noise=None):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image = self.flux_pipeline(
                prompt=prompt,
                guidance_scale=self.guidance_scale,
                height=self.height,
                width=self.width,
                num_inference_steps=self.diffusion_steps,
                max_sequence_length=self.max_sequence_length,
            ).images[0]
        return None, image  # Return None for latents as Flux doesn't expose them