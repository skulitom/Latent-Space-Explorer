import pygame
import asyncio
from collections import deque
from ui_manager import UIManager
from event_handler import EventHandler
from minimap import Minimap
from latent_explorer import LatentSpaceExplorer
import numpy as np

class LatentWalkInterface:
    def __init__(self, cfg, flux_model):
        self.cfg = cfg
        self.explorer = LatentSpaceExplorer(flux_model, cfg)
        self.minimap_panel_width = 250  # Reduced minimap panel width
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
        self.current_vector = np.zeros(2)  # Initialize with a zero vector

    def setup_interface(self):
        pygame.init()
        pygame.freetype.init()
        self.width = self.cfg.get("width", 800)  # Increased window width
        self.height = self.cfg.get("height", 800)  # Increased window height
        self.screen = pygame.display.set_mode((self.width, self.height))  # Remove minimap_panel_width from total width
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
        self.ui_manager.add_log_message("Generating initial image...")
        self.ui_manager.draw()  # Force an immediate update of the display
        pygame.display.flip()
        self.current_image = await self.explorer.generate_image(self.prompt_text)
        if self.current_image:
            self.ui_manager.add_log_message("Initial image generated successfully")
        else:
            self.ui_manager.add_log_message("Failed to generate initial image")

    def update(self):
        if self.task_queue and not self.processing_task:
            self.processing_task = asyncio.create_task(self.process_queue())

    async def process_queue(self):
        while self.task_queue:
            task = self.task_queue.popleft()
            await task
        self.processing_task = None

    async def reset_image(self):
        self.ui_manager.add_log_message("Resetting image...")
        self.ui_manager.draw()  # Force an immediate update of the display
        pygame.display.flip()
        _, reset_prompt = await self.explorer.reset_position()
        self.current_image = await self.update_image(reset_prompt)
        self.ui_manager.add_log_message("Reset completed")
        print("Reset completed")
        self.minimap.reset_position()

    async def move_image(self, move_direction):
        self.ui_manager.add_log_message(f"Moving {move_direction}...")
        self.ui_manager.draw()  # Force an immediate update of the display
        pygame.display.flip()
        _, new_prompt = await self.explorer.update_latents(self.prompt_text, self.direction_text, move_direction, self.cfg['step_size'])
        self.current_image = await self.explorer.generate_image(new_prompt)
        self.ui_manager.add_log_message(f"Move {move_direction} completed")
        self.ui_manager.add_log_message(f"Direction: {self.direction_text}")
        print(f"Move completed successfully. New prompt: {new_prompt}")
        self.minimap.update_position(move_direction)
        self.current_vector = self._update_vector(move_direction)

    async def update_image(self, new_prompt):
        self.ui_manager.add_log_message("Generating new image...")
        self.ui_manager.draw()  # Force an immediate update of the display
        pygame.display.flip()
        image = await self.explorer.generate_image(new_prompt)
        self.ui_manager.add_log_message("Image generated")
        return image

    def _update_vector(self, move_direction):
        step_size = self.cfg['step_size']
        if move_direction == 'forward':
            return self.current_vector + np.array([0, step_size])
        elif move_direction == 'backward':
            return self.current_vector + np.array([0, -step_size])
        elif move_direction == 'left':
            return self.current_vector + np.array([-step_size, 0])
        elif move_direction == 'right':
            return self.current_vector + np.array([step_size, 0])
        return self.current_vector

    def set_direction(self):
        if self.direction_text:
            self.current_vector = self.ui_manager._calculate_direction_vector(self.direction_text)
            self.ui_manager.add_log_message(f"Direction set: {self.direction_text}")
        else:
            self.current_vector = np.zeros(2)
            self.ui_manager.add_log_message("Direction cleared")