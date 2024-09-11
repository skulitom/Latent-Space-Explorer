import pygame
import pygame.freetype
from PIL import Image

class UIManager:
    # UI Constants
    PADDING = 15  # Reduced padding
    INPUT_HEIGHT = 50
    INSTRUCTION_HEIGHT = 30
    CURSOR_BLINK_TIME = 500

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
        self.draw_side_panel()
        self.draw_bottom_panel()
        pygame.display.flip()

    def draw_main_content(self):
        if self.interface.current_image:
            pygame_image = self.pil_image_to_pygame_surface(self.interface.current_image)
            
            # Calculate the size for the main content area
            content_width = self.interface.width - self.interface.minimap_panel_width - 2 * self.PADDING
            bottom_panel_height = self.INPUT_HEIGHT + self.INSTRUCTION_HEIGHT + 2 * self.PADDING
            content_height = self.interface.height - bottom_panel_height - 2 * self.PADDING
            
            image_size = min(content_width, content_height)
            scaled_image = pygame.transform.scale(pygame_image, (image_size, image_size))
            
            # Center the image in the main content area, but closer to the top
            image_rect = scaled_image.get_rect(center=(
                (self.interface.width - self.interface.minimap_panel_width) // 2,
                self.PADDING + image_size // 2
            ))
            self.interface.screen.blit(scaled_image, image_rect)
            pygame.draw.rect(self.interface.screen, self.input_border, image_rect, 1)

    def draw_side_panel(self):
        panel_rect = pygame.Rect(self.interface.width - self.interface.minimap_panel_width, 0,
                                 self.interface.minimap_panel_width, self.interface.height)
        pygame.draw.rect(self.interface.screen, self.input_bg, panel_rect)
        pygame.draw.line(self.interface.screen, self.input_border,
                         (panel_rect.left, 0), (panel_rect.left, self.interface.height), 2)

        # Draw minimap title
        title_font = pygame.font.SysFont("Arial", 20)
        title_surface = title_font.render("Latent Space Map", True, self.text_color)
        title_rect = title_surface.get_rect(center=(panel_rect.centerx, 30))
        self.interface.screen.blit(title_surface, title_rect)

        # Draw minimap
        minimap_size = min(self.interface.minimap_panel_width - 2 * self.PADDING, 
                           panel_rect.height - title_rect.bottom - 3 * self.PADDING)
        minimap_rect = pygame.Rect(panel_rect.left + (panel_rect.width - minimap_size) // 2,
                                   title_rect.bottom + self.PADDING,
                                   minimap_size, minimap_size)
        self.interface.minimap.draw(self.interface.screen, minimap_rect)

    def draw_bottom_panel(self):
        panel_height = self.INPUT_HEIGHT + self.INSTRUCTION_HEIGHT + 2 * self.PADDING
        panel_rect = pygame.Rect(0, self.interface.height - panel_height,
                                 self.interface.width - self.interface.minimap_panel_width, panel_height)
        
        # Draw input box
        input_box = pygame.Rect(self.PADDING, panel_rect.top + self.PADDING,
                                panel_rect.width - 2 * self.PADDING, self.INPUT_HEIGHT)
        
        pygame.draw.rect(self.interface.screen, self.input_bg, input_box)
        pygame.draw.rect(self.interface.screen, self.input_border, input_box, 1)
        text_surface, _ = self.font.render(self.interface.direction_text, self.text_color)
        self.interface.screen.blit(text_surface, (input_box.x + 10, input_box.y + 15))  # Adjusted vertical position

        # Draw blinking cursor
        if self.interface.text_input_active:
            self.cursor_timer += self.interface.clock.get_time()
            if self.cursor_timer >= self.CURSOR_BLINK_TIME:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0

            if self.cursor_visible:
                cursor_x = input_box.x + 10 + text_surface.get_width()
                cursor_y = input_box.y + 15
                pygame.draw.line(self.interface.screen, self.text_color, (cursor_x, cursor_y), (cursor_x, cursor_y + 25), 2)

        # Draw instructions
        instructions = "TAB: Toggle input | WASD: Move | R: Reset | ENTER: Set direction"
        instruction_surface, _ = self.font.render(instructions, self.text_color)
        instruction_rect = instruction_surface.get_rect(center=(panel_rect.centerx, 
                                                                panel_rect.bottom - self.INSTRUCTION_HEIGHT // 2))
        self.interface.screen.blit(instruction_surface, instruction_rect)

    @staticmethod
    def pil_image_to_pygame_surface(pil_image):
        return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()