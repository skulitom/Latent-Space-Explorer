import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import logging
from clip_slider_pipeline import CLIPSliderFlux
from functools import lru_cache

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
            nn.Linear(2048, np.prod(latent_shape), dtype=dtype),
            nn.Tanh()
        )
    
    def forward(self, combined_embed: torch.Tensor) -> torch.Tensor:
        return self.net(combined_embed).view(-1, *self.latent_shape)

class LatentSpaceExplorer:
    def __init__(self, flux_model, config):
        self.flux_model = flux_model
        self.config = config
        self.base_prompt = config.get("prompt", "")
        self.current_prompt = self.base_prompt
        self.direction_strengths = {}
        self.max_strength = 5
        self.clip_slider = CLIPSliderFlux(flux_model, flux_model.device, config)

    async def update_latents(self, prompt_text: str, direction_text: str, 
                             move_direction: str, step_size: float) -> tuple[None, str]:
        if direction_text not in self.direction_strengths:
            self.direction_strengths[direction_text] = 0
            opposite = f"not {direction_text}"
            await self.clip_slider.find_latent_direction(direction_text, opposite)

        if move_direction == 'forward':
            self.direction_strengths[direction_text] = min(self.direction_strengths[direction_text] + step_size, self.max_strength)
        elif move_direction == 'backward':
            self.direction_strengths[direction_text] = max(self.direction_strengths[direction_text] - step_size, -self.max_strength)

        return None, self.base_prompt

    async def reset_position(self):
        self.direction_strengths.clear()
        return None, self.base_prompt

    async def generate_image(self, prompt: str):
        return await self.clip_slider.generate(prompt, self.direction_strengths)