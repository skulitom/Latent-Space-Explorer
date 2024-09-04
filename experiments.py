import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import pygame.freetype
from pygame import gfxdraw
import colorsys
from PIL import Image
import numpy as np
from typing import Tuple, Optional
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
        # Flux doesn't expose latents, so we'll use a random tensor
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
            # Flux doesn't use latent noise, so we'll ignore it
            current_embed = self.flux_model.encode_prompt(prompt_text)
            direction_embed = self.flux_model.encode_prompt(f"{prompt_text}, {direction_text}")

            # Instead of manipulating latents, we'll just return the original noise and the new prompt
            new_prompt = f"{prompt_text}, {direction_text}, {move_direction}"
            return latent_noise, new_prompt

        except Exception as e:
            logger.error(f"Error in update_latents: {e}", exc_info=True)
            return latent_noise, prompt_text

    def precompute_orthogonal_directions(self, num_directions: int = 100) -> torch.Tensor:
        latent_dim = np.prod(self.latent_shape)
        directions = torch.randn(num_directions, latent_dim, device=self.device, dtype=self.model_dtype)
        
        # Gram-Schmidt process for orthogonalization
        for i in range(1, num_directions):
            for j in range(i):
                directions[i] -= torch.dot(directions[i], directions[j]) * directions[j]
            directions[i] = F.normalize(directions[i], dim=0)
        
        return directions

    def get_nearest_orthogonal_direction(self, direction: torch.Tensor) -> torch.Tensor:
        direction = direction.view(-1)  # Flatten the direction tensor
        direction = F.normalize(direction, dim=0)
        
        # Ensure the direction has the same size and dtype as each row in orthogonal_directions
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

def create_latent_walk_interface(cfg_path, cfg, flux_model):
    explorer = LatentSpaceExplorer(flux_model, cfg)
    latent_noise = explorer.initialize_latents()

    pygame.init()
    screen = pygame.display.set_mode((cfg["width"], cfg["height"] + 100), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    pygame.freetype.init()
    font = pygame.freetype.SysFont(None, 24)

    prompt_text = cfg['prompt']
    
    print("Generating initial image...")
    _, current_image = flux_model.run_flux_inference(prompt_text)
    print("Initial image generated")

    pygame_image = pil_image_to_pygame_surface(current_image)
    screen.blit(pygame_image, (0, 0))

    direction_text = ""
    text_input_active = False
    text_cursor_visible = True
    text_cursor_timer = 0

    current_direction = None
    direction_change_time = 0
    direction_display_duration = 1000  # ms
    
    # Load UI colors from config
    ui_colors = cfg.get('ui_colors', {})
    bg_start = ui_colors.get('background_start', (40, 40, 40))
    bg_end = ui_colors.get('background_end', (20, 20, 20))
    input_bg = ui_colors.get('text_input_bg', (60, 60, 60))
    input_border = ui_colors.get('text_input_border', (100, 100, 100))
    text_color = ui_colors.get('text_color', (200, 200, 200))
    instruction_color = ui_colors.get('instruction_color', (150, 150, 150))

    def draw_rounded_rect(surface, rect, color, corner_radius):
        """Draw a rounded rectangle with a clean look"""
        if corner_radius < 0:
            raise ValueError(f"Corner radius ({corner_radius}) must be >= 0")

        rect_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(rect_surface, color, (0, 0, *rect.size), border_radius=corner_radius)
        surface.blit(rect_surface, rect)

    def create_gradient(width, height, start_color, end_color):
        """Create a vertical gradient surface"""
        gradient = pygame.Surface((width, height), pygame.SRCALPHA)
        for y in range(height):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * y / height)
            g = int(start_color[1] + (end_color[1] - start_color[1]) * y / height)
            b = int(start_color[2] + (end_color[2] - start_color[2]) * y / height)
            pygame.draw.line(gradient, (r, g, b), (0, y), (width, y))
        return gradient

    def draw_ui():
        screen_width, screen_height = screen.get_size()
        image_height = screen_height - 120

        # Scale and draw the image
        scaled_image = pygame.transform.scale(pygame_image, (screen_width, image_height))
        screen.blit(scaled_image, (0, 0))

        # Create and draw gradient background for UI
        gradient = create_gradient(screen_width, 120, bg_start, bg_end)
        screen.blit(gradient, (0, image_height))

        # Draw the text input box
        input_box = pygame.Rect(20, image_height + 20, screen_width - 40, 40)
        draw_rounded_rect(screen, input_box, input_bg, 10)
        
        # Draw the border
        pygame.draw.rect(screen, input_border, input_box, 2, border_radius=10)

        # Render the direction text
        text_surface, _ = font.render(direction_text, text_color)
        screen.blit(text_surface, (input_box.x + 10, input_box.y + 10))

        # Draw the text cursor
        if text_input_active and text_cursor_visible:
            cursor_pos = font.get_rect(direction_text[:len(direction_text)])
            pygame.draw.line(screen, text_color,
                             (input_box.x + 10 + cursor_pos.width, input_box.y + 10),
                             (input_box.x + 10 + cursor_pos.width, input_box.y + 30), 2)

        # Draw instructions
        instructions = [
            "TAB: Toggle text input | WASD: Move | R: Reset",
            "Enter: Submit direction"
        ]
        for i, instruction in enumerate(instructions):
            text_surface, _ = font.render(instruction, instruction_color)
            screen.blit(text_surface, (20, image_height + 70 + i * 25))

        # Display current direction
        if current_direction and pygame.time.get_ticks() - direction_change_time < direction_display_duration:
            direction_text_surface, _ = font.render(f"Moving: {current_direction}", text_color)
            text_rect = direction_text_surface.get_rect(center=(screen_width // 2, image_height + 10))
            screen.blit(direction_text_surface, text_rect)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    text_input_active = not text_input_active
                elif text_input_active:
                    if event.key == pygame.K_RETURN:
                        print(f"Direction set to: {direction_text}")
                        text_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        direction_text = direction_text[:-1]
                    else:
                        direction_text += event.unicode

        if not text_input_active:
            keys = pygame.key.get_pressed()
            move_direction = None
            if keys[pygame.K_w]:
                move_direction = 'forward'
            elif keys[pygame.K_s]:
                move_direction = 'backward'
            elif keys[pygame.K_a]:
                move_direction = 'left'
            elif keys[pygame.K_d]:
                move_direction = 'right'
            elif keys[pygame.K_r]:
                print("Resetting to original image")
                _, current_image = flux_model.run_flux_inference(prompt_text)
                pygame_image = pil_image_to_pygame_surface(current_image)
                print("Reset completed")

            if move_direction and direction_text:
                try:
                    print(f"Attempting to move: {move_direction}")
                    new_prompt = f"{prompt_text}, {direction_text}, {move_direction}"
                    _, new_image = flux_model.run_flux_inference(new_prompt)
                    pygame_image = pil_image_to_pygame_surface(new_image)
                    print("Move completed successfully")
                    current_direction = move_direction
                    direction_change_time = pygame.time.get_ticks()
                except Exception as e:
                    print(f"Error during image update: {e}")
                    import traceback
                    traceback.print_exc()

        # Update text cursor blink
        text_cursor_timer += clock.get_time()
        if text_cursor_timer >= 500:
            text_cursor_visible = not text_cursor_visible
            text_cursor_timer = 0

        draw_ui()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

def pil_image_to_pygame_surface(pil_image: Image.Image) -> pygame.Surface:
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()