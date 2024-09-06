import os
import random
import torch
from diffusers import FluxPipeline
from functools import lru_cache
import asyncio
import torch.amp as amp

class FluxModel:
    def __init__(self, model_cfg):
        self.device = torch.device(f"cuda:{model_cfg.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
        self.flux_pipeline = self._load_flux_pipeline(model_cfg["model_id"])
        self.diffusion_steps = model_cfg.get("diffusion_steps", 4)
        self.guidance_scale = model_cfg.get("guidance_scale", 0.)
        self.height = model_cfg.get("height", 1024)
        self.width = model_cfg.get("width", 1024)
        self.max_sequence_length = model_cfg.get("max_sequence_length", 256)
        
        # Enable gradient checkpointing if available
        if hasattr(self.flux_pipeline, 'enable_gradient_checkpointing'):
            self.flux_pipeline.enable_gradient_checkpointing()
        
        # Use torch.compile() for newer GPUs if possible
        self._compile_model()

        self.cache_size = model_cfg.get("cache_size", 100)
        self.scaler = amp.GradScaler()  # Removed 'cuda' argument as it's not needed

    def _load_flux_pipeline(self, model_id):
        print("Loading Flux model...")
        flux_pipeline = FluxPipeline.from_pretrained(model_id, revision='refs/pr/1', torch_dtype=torch.float16)
        flux_pipeline.enable_sequential_cpu_offload()
        if hasattr(flux_pipeline, 'vae'):
            flux_pipeline.vae.enable_slicing()
            flux_pipeline.vae.enable_tiling()
        else:
            print("Flux pipeline doesn't have a separate VAE component")
        print("Flux model loaded\n")
        return flux_pipeline

    def _compile_model(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            try:
                if hasattr(self.flux_pipeline, 'model'):
                    self.flux_pipeline.model = torch.compile(self.flux_pipeline.model, mode='reduce-overhead')
                    print("Flux model compiled successfully")
                else:
                    print("Flux pipeline doesn't have a 'model' attribute to compile")
            except Exception as e:
                print(f"Failed to compile Flux model: {e}")

    @torch.no_grad()
    def encode_prompt(self, prompt):
        # Flux doesn't have a separate encoding step, so we'll return the prompt as is
        return prompt

    def get_expected_latent_shape(self):
        # Adjust this based on your Flux model's requirements
        return (1, 64, 6144)  # This shape seems to match what the model expects

    @torch.no_grad()
    def run_flux_inference(self, prompt, latent_noise=None):
        try:
            return None, self._cached_inference(prompt, self.height, self.width, self.diffusion_steps, self.guidance_scale)
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None

    @lru_cache(maxsize=100)
    def _cached_inference(self, prompt, height, width, steps, guidance_scale):
        with amp.autocast('cuda', dtype=torch.float16):
            try:
                image = self.flux_pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    max_sequence_length=self.max_sequence_length,
                ).images[0]
                return image
            except Exception as e:
                print(f"Error in _cached_inference: {e}")
                return None

    async def run_flux_inference_async(self, prompt, latent_noise=None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_flux_inference, prompt, latent_noise)