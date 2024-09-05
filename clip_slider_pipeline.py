import torch
from tqdm import tqdm
from PIL import Image

class CLIPSliderFlux:
    def __init__(self, flux_model, device: torch.device, config):
        self.device = device
        self.flux_model = flux_model
        self.direction_vectors = {}
        self.config = config

    async def find_latent_direction(self, direction: str, opposite: str):
        if direction in self.direction_vectors:
            return self.direction_vectors[direction]

        with torch.no_grad():
            pos_prompt = f"a {direction}"
            neg_prompt = f"a {opposite}"

            pos_toks = self.flux_model.flux_pipeline.tokenizer_2(pos_prompt,
                                             return_tensors="pt",
                                             padding="max_length",
                                             truncation=True,
                                             max_length=self.flux_model.flux_pipeline.tokenizer_2.model_max_length).input_ids.to(self.device)
            neg_toks = self.flux_model.flux_pipeline.tokenizer_2(neg_prompt,
                                             return_tensors="pt",
                                             padding="max_length",
                                             truncation=True,
                                             max_length=self.flux_model.flux_pipeline.tokenizer_2.model_max_length).input_ids.to(self.device)
            pos = self.flux_model.flux_pipeline.text_encoder_2(pos_toks, output_hidden_states=False)[0]
            neg = self.flux_model.flux_pipeline.text_encoder_2(neg_toks, output_hidden_states=False)[0]

        diff = pos - neg
        self.direction_vectors[direction] = diff
        return diff

    async def generate(self, prompt: str, directions: dict, **pipeline_kwargs):
        with torch.no_grad():
            text_inputs = self.flux_model.flux_pipeline.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            text_input_ids = text_inputs.input_ids.to(self.device)
            prompt_embeds = self.flux_model.flux_pipeline.text_encoder(text_input_ids, output_hidden_states=False)
            pooled_prompt_embeds = prompt_embeds.pooler_output

            text_inputs = self.flux_model.flux_pipeline.tokenizer_2(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            toks = text_inputs.input_ids.to(self.device)
            prompt_embeds = self.flux_model.flux_pipeline.text_encoder_2(toks, output_hidden_states=False)[0]

            for direction, scale in directions.items():
                if direction in self.direction_vectors:
                    prompt_embeds += self.direction_vectors[direction] * scale

            images = self.flux_model.flux_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=self.config['diffusion_steps'],
                guidance_scale=self.config['guidance_scale'],
                **pipeline_kwargs
            ).images

        return images[0]

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
            image = await self.generate(prompt, {direction: scale}, seed, **pipeline_kwargs)
            images.append(image.resize((512,512)))

        canvas = Image.new('RGB', (640 * steps, 640))
        for i, im in enumerate(images):
            canvas.paste(im, (640 * i, 0))

        return canvas