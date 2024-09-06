import pygame
import pygame.freetype
from PIL import Image
from latent_explorer import LatentSpaceExplorer
import asyncio
from collections import deque
import sys
import traceback

class LatentWalkInterface:
    def __init__(self, cfg, flux_model):
        self.cfg = cfg
        self.explorer = LatentSpaceExplorer(flux_model, cfg)
        self.width = cfg.get("width", 800)
        self.height = cfg.get("height", 600)
        self.ui_colors = cfg.get('ui_colors', {})
        self.setup_pygame()
        self.setup_ui_elements()
        self.task_queue = deque()
        self.processing_task = None
        self.running = True
        self.current_image = None
        self.prompt_text = cfg['prompt']
        self.direction_text = ""
        self.text_input_active = False
        self.text_cursor_visible = True
        self.text_cursor_timer = 0
        self.current_direction = None
        self.direction_change_time = 0
        self.direction_display_duration = 1000  # ms
        self.loading = False
        self.loading_dots = 0
        self.loading_timer = 0

    def setup_pygame(self):
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.display.set_mode((self.width, self.height + 100), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.setup_window()

    def setup_window(self):
        pygame.display.set_caption("Latent Space Explorer")
        icon = pygame.Surface((32, 32), pygame.SRCALPHA)
        accent_color = self.ui_colors.get('accent_color', (0, 120, 215))
        background_color = self.ui_colors.get('background', (18, 18, 18))
        pygame.draw.circle(icon, accent_color, (16, 16), 14)
        pygame.draw.circle(icon, background_color, (16, 16), 10)
        pygame.display.set_icon(icon)

    def setup_ui_elements(self):
        self.font = pygame.freetype.SysFont("Arial", 18)
        self.bg_color = self.ui_colors.get('background', (18, 18, 18))
        self.text_color = self.ui_colors.get('text_color', (230, 230, 230))
        self.accent_color = self.ui_colors.get('accent_color', (0, 120, 215))
        self.input_bg = self.ui_colors.get('text_input_bg', (30, 30, 30))
        self.input_border = self.ui_colors.get('text_input_border', (45, 45, 45))
        self.instruction_color = self.ui_colors.get('instruction_color', (180, 180, 180))

    async def run(self):
        await self.generate_initial_image()
        while self.running:
            try:
                await self.handle_events()
                self.update()
                self.draw()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                traceback.print_exc()
                self.running = False
        pygame.quit()
        print("Game closed.")

    async def generate_initial_image(self):
        print("Generating initial image...")
        self.current_image = await self.explorer.generate_image(self.prompt_text)
        if self.current_image:
            print("Initial image generated successfully")
        else:
            print("Failed to generate initial image")

    async def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.text_input_active = not self.text_input_active
                elif self.text_input_active:
                    if event.key == pygame.K_RETURN:
                        print(f"Direction set to: {self.direction_text}")
                        await self.explorer.update_latents(self.prompt_text, self.direction_text, 'forward', 0)
                        self.text_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.direction_text = self.direction_text[:-1]
                    else:
                        self.direction_text += event.unicode
                elif event.key == pygame.K_r:
                    print("Resetting to original image")
                    self.task_queue.append(self.reset_image())
                else:
                    move_direction = None
                    if event.key == pygame.K_w:
                        move_direction = 'forward'
                    elif event.key == pygame.K_s:
                        move_direction = 'backward'
                    elif event.key == pygame.K_a:
                        move_direction = 'left'
                    elif event.key == pygame.K_d:
                        move_direction = 'right'
                    
                    if move_direction and self.direction_text and not self.loading:
                        self.task_queue.append(self.move_image(move_direction))

    def update(self):
        if self.task_queue and not self.processing_task:
            self.processing_task = asyncio.create_task(self.process_queue())
        
        self.text_cursor_timer += self.clock.get_time()
        if self.text_cursor_timer >= 500:
            self.text_cursor_visible = not self.text_cursor_visible
            self.text_cursor_timer = 0

        if self.loading:
            self.loading_timer += self.clock.get_time()
            if self.loading_timer >= 500:
                self.loading_dots = (self.loading_dots + 1) % 4
                self.loading_timer = 0

    def draw(self):
        self.screen.fill(self.bg_color)
        if self.current_image:
            pygame_image = self.pil_image_to_pygame_surface(self.current_image)
            scaled_image = pygame.transform.scale(pygame_image, (self.width - 40, self.height - 20))
            image_rect = scaled_image.get_rect(center=(self.width // 2, (self.height - 20) // 2 + 10))
            self.screen.blit(scaled_image, image_rect)
            pygame.draw.rect(self.screen, self.input_border, image_rect, 1)

        input_box = pygame.Rect(20, self.height + 10, self.width - 40, 40)
        pygame.draw.rect(self.screen, self.input_bg, input_box)
        pygame.draw.rect(self.screen, self.input_border, input_box, 1)

        text_surface, _ = self.font.render(self.direction_text, self.text_color)
        self.screen.blit(text_surface, (input_box.x + 15, input_box.y + 10))

        if self.text_input_active and self.text_cursor_visible:
            cursor_pos = self.font.get_rect(self.direction_text[:len(self.direction_text)])
            pygame.draw.line(self.screen, self.text_color,
                             (input_box.x + 15 + cursor_pos.width, input_box.y + 8),
                             (input_box.x + 15 + cursor_pos.width, input_box.y + 32), 2)

        instructions = [
            "TAB: Toggle input | WASD: Move | R: Reset | ENTER: Set direction"
        ]
        instruction_surface, _ = self.font.render(instructions[0], self.instruction_color)
        instruction_rect = instruction_surface.get_rect(center=(self.width // 2, self.height + 75))
        self.screen.blit(instruction_surface, instruction_rect)

        if self.current_direction and pygame.time.get_ticks() - self.direction_change_time < self.direction_display_duration:
            direction_text_surface, _ = self.font.render(f"Moving: {self.current_direction}", self.accent_color)
            text_rect = direction_text_surface.get_rect(center=(self.width // 2, self.height + 70))
            self.screen.blit(direction_text_surface, text_rect)

        if self.loading:
            loading_text = f"Processing{'.' * self.loading_dots}"
            loading_surface, _ = self.font.render(loading_text, self.accent_color)
            loading_rect = loading_surface.get_rect(center=(self.width // 2, self.height + 70))
            self.screen.blit(loading_surface, loading_rect)

        pygame.display.flip()

    @staticmethod
    def pil_image_to_pygame_surface(pil_image):
        return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()

    async def process_queue(self):
        while self.task_queue:
            task = self.task_queue.popleft()
            await task
        self.processing_task = None

    async def reset_image(self):
        _, reset_prompt = await self.explorer.reset_position()
        self.current_image = await self.update_image(reset_prompt)
        print("Reset completed")

    async def move_image(self, move_direction):
        self.loading = True
        _, new_prompt = await self.explorer.update_latents(self.prompt_text, self.direction_text, move_direction, self.cfg['step_size'])
        
        # Generate a single image for the new position
        self.current_image = await self.explorer.generate_image(new_prompt)
        print(f"Generated image for move: {move_direction}")
        print(f"Image size: {self.current_image.size}, Mode: {self.current_image.mode}")
        self.draw()

        print(f"Move completed successfully. New prompt: {new_prompt}")
        self.current_direction = move_direction
        self.direction_change_time = pygame.time.get_ticks()
        self.loading = False

    async def update_image(self, new_prompt):
        return await self.explorer.generate_image(new_prompt)

async def create_latent_walk_interface(cfg_path, cfg, flux_model):
    interface = LatentWalkInterface(cfg, flux_model)
    await interface.run()