import torch
from PIL import Image
from functools import lru_cache
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import numpy as np

class CLIPSliderFlux:
    def __init__(self, flux_model, device: torch.device, config):
        self.device = device
        self.flux_model = flux_model
        self.direction_vectors = {}
        self.config = config
        print(f"Initialized CLIPSliderFlux with config: {self.config}")
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed

    @lru_cache(maxsize=100)
    def _cached_text_encoding(self, text):
        with torch.no_grad():
            toks = self.flux_model.flux_pipeline.tokenizer_2(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.flux_model.flux_pipeline.tokenizer_2.model_max_length
            ).input_ids.to(self.device)
            return self.flux_model.flux_pipeline.text_encoder_2(toks, output_hidden_states=False)[0]

    async def find_latent_direction(self, direction: str, opposite: str):
        if direction in self.direction_vectors:
            return self.direction_vectors[direction]

        pos = self._cached_text_encoding(f"a {direction}")
        neg = self._cached_text_encoding(f"a {opposite}")

        diff = pos - neg
        self.direction_vectors[direction] = diff
        return diff

    async def generate_image(self, prompt: str, directions: dict, seed: int = None):
        with torch.no_grad():
            prompt_embeds = self._cached_text_encoding(prompt)
            pooled_prompt_embeds = self._get_pooled_prompt_embeds(prompt)

            for direction, scale in directions.items():
                if direction in self.direction_vectors:
                    prompt_embeds += self.direction_vectors[direction] * scale

            if seed is not None:
                torch.manual_seed(seed)
            
            image = self.flux_model.flux_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=self.config['diffusion_steps'],
                guidance_scale=self.config['guidance_scale'],
                height=800,  # Set height to half of the default (1024)
                width=800,   # Set width to half of the default (1024)
            ).images[0]

            return image

    def _get_pooled_prompt_embeds(self, prompt: str):
        return self.flux_model.flux_pipeline.text_encoder(
            self.flux_model.flux_pipeline.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=77
            ).input_ids.to(self.device),
            output_hidden_states=False
        ).pooler_output
    async def spectrum(self,
                 prompt: str,
                 direction: str,
                 low_scale: float = -2,
                 high_scale: float = 2,
                 steps: int = 5,
                 seed: int = 15,
                 **pipeline_kwargs
                 ):

        images = []
        for i in range(steps):
            scale = low_scale + (high_scale - low_scale) * i / (steps - 1)
            image = await self.generate_image(prompt, {direction: scale})
            images.append(image.resize((512,512)))

        canvas = Image.new('RGB', (640 * steps, 640))
        for i, im in enumerate(images):
            canvas.paste(im, (640 * i, 0))

        return canvas

    async def generate_batch(self, prompts: List[str], directions: List[Dict[str, float]], **pipeline_kwargs):
        with torch.no_grad():
            prompt_embeds = torch.cat([self._cached_text_encoding(prompt) for prompt in prompts])
            pooled_prompt_embeds = self.flux_model.flux_pipeline.text_encoder(
                self.flux_model.flux_pipeline.tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(self.device),
                output_hidden_states=False
            ).pooler_output

            for i, direction_dict in enumerate(directions):
                for direction, scale in direction_dict.items():
                    if direction in self.direction_vectors:
                        prompt_embeds[i] += self.direction_vectors[direction] * scale

            images = self.flux_model.flux_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=self.config['diffusion_steps'],
                guidance_scale=self.config['guidance_scale'],
                **pipeline_kwargs
            ).images

        return images

    def __del__(self):
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
