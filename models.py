import os
import random
import torch
from diffusers import StableDiffusionPipeline, FluxPipeline
from torchvision import transforms

class StableDiffusion:
    def __init__(self, model_cfg):
        self.device = torch.device(f"cuda:{model_cfg.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
        self.sd_pipeline = self._load_sd_pipeline(model_cfg["model_id"])
        self.diffusion_steps = model_cfg.get("diffusion_steps", 50)
        self.guidance_scale = model_cfg.get("guidance_scale", 7.5)
        self.height = model_cfg.get("height")
        self.width = model_cfg.get("width")
        self.rand_seed = model_cfg.get("rand_seed", random.randint(0, 10**6))

    def _load_sd_pipeline(self, model_id):
        print("Loading Stable Diffusion model from the diffusers library...")
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        print("Stable Diffusion model loaded\n")
        return sd_pipeline

    def sample_noise(self, height, width, images_per_prompt, rand_seed):
        scale_factor = 2 ** (len(self.sd_pipeline.vae.config.block_out_channels) - 1)
        shape = (4, height // scale_factor, width // scale_factor)  # Remove batch dimension
        
        # Create the generator on the correct device
        generator = torch.Generator(device=self.device).manual_seed(rand_seed)
        
        # Generate the noise directly on the correct device
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        return self.sd_pipeline._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None
        )

    @torch.no_grad()
    def decode_images(self, images):
        transform = transforms.ToPILImage()
        images = self.sd_pipeline.vae.decode(images).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return [transform(images[0].float())]

    @torch.no_grad()
    def run_sd_inference(self, prompt_embed, latent_noise, image=None, mask=None):
        self.sd_pipeline.scheduler.set_timesteps(self.diffusion_steps, device=self.device)
        latent_noise = latent_noise * self.sd_pipeline.scheduler.init_noise_sigma

        # Add batch dimension if it's missing
        if latent_noise.dim() == 3:
            latent_noise = latent_noise.unsqueeze(0)

        for t in self.sd_pipeline.scheduler.timesteps:
            latent_model_input = torch.cat([latent_noise] * 2)
            latent_model_input = self.sd_pipeline.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.sd_pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embed).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent_noise = self.sd_pipeline.scheduler.step(noise_pred, t, latent_noise).prev_sample

        img_embeds = latent_noise / self.sd_pipeline.vae.config.scaling_factor
        image = self.decode_images(img_embeds)[0]  # Get the first (and only) image

        return img_embeds, image

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

    # Implement other methods as needed, similar to StableDiffusion class
    # ...