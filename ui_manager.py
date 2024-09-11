import pygame
import pygame.freetype
from PIL import Image

class UIManager:
    # UI Constants
    PADDING = 20
    INPUT_HEIGHT = 40
    INSTRUCTION_HEIGHT = 30
    CURSOR_BLINK_TIME = 500  # Cursor blinks every 500ms

    def __init__(self, interface):
        self.interface = interface
        self.setup_ui_elements()
        self.cursor_visible = True
        self.cursor_timer = 0

    def setup_ui_elements(self):
        self.font = pygame.freetype.SysFont("Arial", 18)
        self.bg_color = self.interface.cfg.get('ui_colors', {}).get('background', (18, 18, 18))
        self.text_color = self.interface.cfg.get('ui_colors', {}).get('text_color', (230, 230, 230))
        self.accent_color = self.interface.cfg.get('ui_colors', {}).get('accent_color', (0, 120, 215))
        self.input_bg = self.interface.cfg.get('ui_colors', {}).get('text_input_bg', (30, 30, 30))
        self.input_border = self.interface.cfg.get('ui_colors', {}).get('text_input_border', (45, 45, 45))

    def draw(self):
        self.interface.screen.fill(self.bg_color)
        self.draw_main_content()
        self.draw_input_box()
        self.draw_instructions()
        self.draw_map_panel()
        self.interface.minimap.draw(self.interface.screen)
        pygame.display.flip()

    def draw_main_content(self):
        if self.interface.current_image:
            pygame_image = self.pil_image_to_pygame_surface(self.interface.current_image)
            
            # Calculate the size for a square image, using most of the available width
            image_size = min(self.interface.width - 2 * self.PADDING, 
                             self.interface.height - 2 * self.PADDING - self.INPUT_HEIGHT - self.INSTRUCTION_HEIGHT)
            scaled_image = pygame.transform.scale(pygame_image, (image_size, image_size))
            
            # Center the image
            image_rect = scaled_image.get_rect(center=(self.interface.width // 2, 
                                                       (self.interface.height - self.INPUT_HEIGHT - self.INSTRUCTION_HEIGHT) // 2))
            self.interface.screen.blit(scaled_image, image_rect)
            pygame.draw.rect(self.interface.screen, self.input_border, image_rect, 1)

    def draw_input_box(self):
        # Calculate input box width based on the image size
        image_size = min(self.interface.width - 2 * self.PADDING, 
                         self.interface.height - 2 * self.PADDING - self.INPUT_HEIGHT - self.INSTRUCTION_HEIGHT)
        input_box = pygame.Rect((self.interface.width - image_size) // 2, 
                                self.interface.height - self.INPUT_HEIGHT - self.INSTRUCTION_HEIGHT, 
                                image_size, self.INPUT_HEIGHT)
        
        pygame.draw.rect(self.interface.screen, self.input_bg, input_box)
        pygame.draw.rect(self.interface.screen, self.input_border, input_box, 1)
        text_surface, _ = self.font.render(self.interface.direction_text, self.text_color)
        self.interface.screen.blit(text_surface, (input_box.x + 15, input_box.y + 10))

        # Draw blinking cursor
        if self.interface.text_input_active:
            self.cursor_timer += self.interface.clock.get_time()
            if self.cursor_timer >= self.CURSOR_BLINK_TIME:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0

            if self.cursor_visible:
                cursor_x = input_box.x + 15 + text_surface.get_width()
                cursor_y = input_box.y + 10
                pygame.draw.line(self.interface.screen, self.text_color, (cursor_x, cursor_y), (cursor_x, cursor_y + 20), 2)

    def draw_instructions(self):
        instructions = "TAB: Toggle input | WASD: Move | R: Reset | ENTER: Set direction"
        instruction_surface, _ = self.font.render(instructions, self.text_color)
        instruction_rect = instruction_surface.get_rect(center=(self.interface.width // 2, 
                                                                self.interface.height - self.INSTRUCTION_HEIGHT // 2))
        self.interface.screen.blit(instruction_surface, instruction_rect)

    def draw_map_panel(self):
        map_panel_rect = pygame.Rect(self.interface.width, 0, self.interface.minimap_panel_width, self.interface.height + 100)
        pygame.draw.rect(self.interface.screen, self.input_bg, map_panel_rect)
        pygame.draw.line(self.interface.screen, self.input_border, (self.interface.width, 0), (self.interface.width, self.interface.height + 100), 2)

    @staticmethod
    def pil_image_to_pygame_surface(pil_image):
        return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()