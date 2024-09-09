import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import logging
from clip_slider_pipeline import CLIPSliderFlux
from functools import lru_cache
import asyncio
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionMapper(nn.Module):
    def __init__(self, input_dim: int, latent_shape: Tuple[int, ...], dtype=torch.float32):
        super().__init__()
        self.latent_shape = latent_shape
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048, dtype=dtype),
            nn.ReLU(),
            nn.Linear(2048, 2048, dtype=dtype),
            nn.ReLU(),
            nn.Linear(2048, int(np.prod(latent_shape)), dtype=dtype),
            nn.Tanh()
        )
    
    def forward(self, combined_embed: torch.Tensor) -> torch.Tensor:
        return self.net(combined_embed).view(-1, *self.latent_shape)

class LatentSpaceExplorer:
    def __init__(self, flux_model, config):
        self.flux_model = flux_model
        self.config = config
        self.base_prompt = config.get("prompt", "A beautiful Fantasy Landscape")
        self.direction_strengths = {}
        self.max_strength = 5
        self.clip_slider = CLIPSliderFlux(flux_model, flux_model.device, config)
        self.seed = config.get("rand_seed", 0)  # Get seed from config or use a default
        print(f"LatentSpaceExplorer initialized with config: {config}")

    async def update_latents(self, prompt_text: str, direction_text: str, 
                             move_direction: str, step_size: float) -> tuple[None, str]:
        if direction_text not in self.direction_strengths:
            self.direction_strengths[direction_text] = 0
            await self.clip_slider.find_latent_direction(direction_text, f"not {direction_text}")

        old_strength = self.direction_strengths[direction_text]
        self.direction_strengths[direction_text] = self._update_strength(
            old_strength, move_direction, step_size
        )
        
        return None, self.base_prompt

    def _update_strength(self, current_strength: float, move_direction: str, step_size: float) -> float:
        if move_direction == 'forward':
            return min(current_strength + step_size, self.max_strength)
        elif move_direction == 'backward':
            return max(current_strength - step_size, -self.max_strength)
        return current_strength

    async def reset_position(self):
        self.direction_strengths.clear()
        return None, self.base_prompt

    async def generate_image(self, prompt: str):
        try:
            print(f"Generating image with config: diffusion_steps={self.config['diffusion_steps']}, guidance_scale={self.config['guidance_scale']}")
            image = await self.clip_slider.generate_image(self.base_prompt, self.direction_strengths, seed=self.seed)
            print("Base prompt: ", self.base_prompt)
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            traceback.print_exc()
            return None