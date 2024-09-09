import pygame
import asyncio
from collections import deque
from ui_manager import UIManager
from event_handler import EventHandler
from minimap import Minimap
from latent_explorer import LatentSpaceExplorer

class LatentWalkInterface:
    def __init__(self, cfg, flux_model):
        self.cfg = cfg
        self.explorer = LatentSpaceExplorer(flux_model, cfg)
        self.setup_interface()
        self.ui_manager = UIManager(self)
        self.event_handler = EventHandler(self)
        self.minimap = Minimap(self)
        self.task_queue = deque()
        self.processing_task = None
        self.running = True
        self.current_image = None
        self.prompt_text = cfg['prompt']
        self.direction_text = ""
        self.text_input_active = False  # Add this line
        self.clock = pygame.time.Clock()

    def setup_interface(self):
        pygame.init()
        pygame.freetype.init()
        self.width = self.cfg.get("width", 800)
        self.height = self.cfg.get("height", 600)
        self.minimap_panel_width = 200
        self.total_width = self.width + self.minimap_panel_width
        self.screen = pygame.display.set_mode((self.total_width, self.height + 100), pygame.RESIZABLE)
        pygame.display.set_caption("Latent Space Explorer")

    async def run(self):
        await self.generate_initial_image()
        while self.running:
            try:
                await self.event_handler.handle_events()
                self.update()
                self.ui_manager.draw()
                self.clock.tick(60)  # Limit the frame rate to 60 FPS
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
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

    def update(self):
        if self.task_queue and not self.processing_task:
            self.processing_task = asyncio.create_task(self.process_queue())

    async def process_queue(self):
        while self.task_queue:
            task = self.task_queue.popleft()
            await task
        self.processing_task = None

    async def reset_image(self):
        _, reset_prompt = await self.explorer.reset_position()
        self.current_image = await self.update_image(reset_prompt)
        print("Reset completed")
        self.minimap.reset_position()

    async def move_image(self, move_direction):
        _, new_prompt = await self.explorer.update_latents(self.prompt_text, self.direction_text, move_direction, self.cfg['step_size'])
        self.current_image = await self.explorer.generate_image(new_prompt)
        print(f"Move completed successfully. New prompt: {new_prompt}")
        self.minimap.update_position(move_direction)

    async def update_image(self, new_prompt):
        return await self.explorer.generate_image(new_prompt)