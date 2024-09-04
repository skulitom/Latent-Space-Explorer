import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionMapper(nn.Module):
    def __init__(self, input_dim: int, latent_shape: Tuple[int, ...]):
        super().__init__()
        self.latent_shape = latent_shape
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, np.prod(latent_shape)),
            nn.Tanh()
        )
    
    def forward(self, current_embed: torch.Tensor, direction_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current_embed, direction_embed], dim=-1)
        return self.net(combined).view(-1, *self.latent_shape)

class LatentSpaceExplorer:
    def __init__(self, flux_model, config):
        self.flux_model = flux_model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = torch.float16
        self.direction_mapper = None
        self.update_count = 0
        self.reset_interval = 10
        self.original_latent_noise = None
        self.orthogonal_directions = None
        self.latent_shape = None
        self.initialize_latents()

    def initialize_latents(self):
        self.original_latent_noise = torch.randn(
            (1, 4, self.config["height"] // 8, self.config["width"] // 8),
            device=self.device, dtype=self.model_dtype
        )
        self.latent_shape = self.original_latent_noise.shape[1:]
        self.orthogonal_directions = self.precompute_orthogonal_directions()
        return self.original_latent_noise.clone()

    def update_latents(self, latent_noise: torch.Tensor, prompt_text: str, direction_text: str, 
                       move_direction: str, step_size: float) -> Tuple[torch.Tensor, str]:
        try:
            new_prompt = f"{prompt_text}, {direction_text}, {move_direction}"
            return latent_noise, new_prompt
        except Exception as e:
            logger.error(f"Error in update_latents: {e}", exc_info=True)
            return latent_noise, prompt_text

    def precompute_orthogonal_directions(self, num_directions: int = 100) -> torch.Tensor:
        latent_dim = np.prod(self.latent_shape)
        directions = torch.randn(num_directions, latent_dim, device=self.device, dtype=self.model_dtype)
        
        for i in range(1, num_directions):
            for j in range(i):
                directions[i] -= torch.dot(directions[i], directions[j]) * directions[j]
            directions[i] = F.normalize(directions[i], dim=0)
        
        return directions

    def get_nearest_orthogonal_direction(self, direction: torch.Tensor) -> torch.Tensor:
        direction = direction.view(-1)
        direction = F.normalize(direction, dim=0)
        
        if direction.size(0) != self.orthogonal_directions.size(1) or direction.dtype != self.orthogonal_directions.dtype:
            logger.warning(f"Direction size ({direction.size(0)}) or dtype ({direction.dtype}) doesn't match orthogonal directions ({self.orthogonal_directions.size(1)}, {self.orthogonal_directions.dtype}). Adjusting...")
            direction = direction.to(dtype=self.orthogonal_directions.dtype)
            if direction.size(0) > self.orthogonal_directions.size(1):
                direction = direction[:self.orthogonal_directions.size(1)]
            else:
                direction = F.pad(direction, (0, self.orthogonal_directions.size(1) - direction.size(0)))
        
        similarities = torch.matmul(self.orthogonal_directions, direction)
        nearest_index = torch.argmax(similarities)
        return self.orthogonal_directions[nearest_index].view(-1, *self.latent_shape)

    @staticmethod
    def get_scaled_diff(diff_embed: torch.Tensor, perpendicular_embed: torch.Tensor, 
                        move_direction: str, step_size: float) -> torch.Tensor:
        if move_direction == 'forward':
            return diff_embed * step_size
        elif move_direction == 'backward':
            return -diff_embed * step_size
        elif move_direction == 'left':
            return -perpendicular_embed * step_size
        elif move_direction == 'right':
            return perpendicular_embed * step_size
        else:
            return torch.zeros_like(diff_embed)

    def guided_diffusion(self, latent_noise: torch.Tensor, text_embed: torch.Tensor, steps: int = 10) -> torch.Tensor:
        original_steps = self.flux_model.diffusion_steps
        self.flux_model.diffusion_steps = steps
        try:
            with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                result = self.flux_model.run_flux_inference(text_embed, latent_noise)
                latents = result[0] if isinstance(result, tuple) and len(result) > 0 else latent_noise
        except Exception as e:
            logger.error(f"Error during diffusion: {e}", exc_info=True)
            latents = latent_noise
        finally:
            self.flux_model.diffusion_steps = original_steps
        return latents

    @staticmethod
    def adaptive_interpolation(latent_noise: torch.Tensor, new_latents: torch.Tensor, 
                               scaled_diff: torch.Tensor, current_embed: torch.Tensor) -> torch.Tensor:
        factor = torch.tanh(torch.norm(scaled_diff) / torch.norm(current_embed))
        smoothstep_factor = factor * factor * (3 - 2 * factor)
        return (1 - smoothstep_factor) * latent_noise + smoothstep_factor * new_latents