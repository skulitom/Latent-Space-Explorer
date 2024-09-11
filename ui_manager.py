import pygame
import pygame.freetype
from PIL import Image

class UIManager:
    # UI Constants
    PADDING = 15
    INPUT_HEIGHT = 50
    INSTRUCTION_HEIGHT = 30
    CURSOR_BLINK_TIME = 500
    FONT_SIZE = 18
    TITLE_FONT_SIZE = 20
    FONT_NAME = "Arial"
    CURSOR_WIDTH = 2
    BORDER_WIDTH = 1
    PANEL_BORDER_WIDTH = 2
    TEXT_VERTICAL_OFFSET = 15
    CURSOR_HEIGHT = 25

    def __init__(self, interface):
        self.interface = interface
        self.setup_ui_elements()
        self.cursor_visible = True
        self.cursor_timer = 0
        self.log_messages = []  # Add this line to store log messages
        self.max_log_messages = 7  # Increased number of visible log messages
        self.log_bg_color = (25, 25, 25)  # Darker background for log container

    def setup_ui_elements(self):
        self.font = pygame.freetype.SysFont(self.FONT_NAME, self.FONT_SIZE)
        self.title_font = pygame.font.SysFont(self.FONT_NAME, self.TITLE_FONT_SIZE)
        self.colors = self._load_colors()

    def _load_colors(self):
        return {
            'bg': self.interface.cfg.get('ui_colors', {}).get('background', (18, 18, 18)),
            'text': self.interface.cfg.get('ui_colors', {}).get('text_color', (230, 230, 230)),
            'accent': self.interface.cfg.get('ui_colors', {}).get('accent_color', (0, 120, 215)),
            'input_bg': self.interface.cfg.get('ui_colors', {}).get('text_input_bg', (30, 30, 30)),
            'input_border': self.interface.cfg.get('ui_colors', {}).get('text_input_border', (45, 45, 45))
        }

    def draw(self):
        self.interface.screen.fill(self.colors['bg'])
        self.draw_main_content()
        self.draw_side_panel()
        self.draw_bottom_panel()
        pygame.display.flip()

    def draw_main_content(self):
        if self.interface.current_image:
            pygame_image = self.pil_image_to_pygame_surface(self.interface.current_image)
            
            content_width = self.interface.width - self.interface.minimap_panel_width - 2 * self.PADDING
            bottom_panel_height = self.INPUT_HEIGHT + self.INSTRUCTION_HEIGHT + 2 * self.PADDING
            content_height = self.interface.height - bottom_panel_height - 2 * self.PADDING
            
            image_size = min(content_width, content_height)
            scaled_image = pygame.transform.scale(pygame_image, (image_size, image_size))
            
            image_rect = scaled_image.get_rect(center=(
                (self.interface.width - self.interface.minimap_panel_width) // 2,
                self.PADDING + image_size // 2
            ))
            self.interface.screen.blit(scaled_image, image_rect)
            pygame.draw.rect(self.interface.screen, self.colors['input_border'], image_rect, self.BORDER_WIDTH)

    def draw_side_panel(self):
        panel_rect = self._get_side_panel_rect()
        pygame.draw.rect(self.interface.screen, self.colors['input_bg'], panel_rect)
        pygame.draw.line(self.interface.screen, self.colors['input_border'],
                         (panel_rect.left, 0), (panel_rect.left, self.interface.height), self.PANEL_BORDER_WIDTH)

        self._draw_minimap_title(panel_rect)
        minimap_rect = self._draw_minimap(panel_rect)
        self._draw_log_display(panel_rect, minimap_rect)

    def _get_side_panel_rect(self):
        return pygame.Rect(self.interface.width - self.interface.minimap_panel_width, 0,
                           self.interface.minimap_panel_width, self.interface.height)

    def _draw_minimap_title(self, panel_rect):
        title_surface = self.title_font.render("Latent Space Map", True, self.colors['text'])
        title_rect = title_surface.get_rect(center=(panel_rect.centerx, 30))
        self.interface.screen.blit(title_surface, title_rect)
        return title_rect

    def _draw_minimap(self, panel_rect):
        title_rect = self._draw_minimap_title(panel_rect)
        minimap_size = min(self.interface.minimap_panel_width - 2 * self.PADDING, 
                           panel_rect.height - title_rect.bottom - 3 * self.PADDING - 100)  # Reduced size to make room for log
        minimap_rect = pygame.Rect(panel_rect.left + (panel_rect.width - minimap_size) // 2,
                                   title_rect.bottom + self.PADDING,
                                   minimap_size, minimap_size)
        self.interface.minimap.draw(self.interface.screen, minimap_rect)
        return minimap_rect

    def _draw_log_display(self, panel_rect, minimap_rect):
        log_title_surface = self.title_font.render("Log", True, self.colors['text'])
        log_title_rect = log_title_surface.get_rect(topleft=(panel_rect.left + self.PADDING, minimap_rect.bottom + self.PADDING))
        self.interface.screen.blit(log_title_surface, log_title_rect)

        log_container_rect = pygame.Rect(
            panel_rect.left + self.PADDING,
            log_title_rect.bottom + 5,
            panel_rect.width - 2 * self.PADDING,
            self.FONT_SIZE * self.max_log_messages + 20  # Increased height
        )
        pygame.draw.rect(self.interface.screen, self.log_bg_color, log_container_rect)
        pygame.draw.rect(self.interface.screen, self.colors['input_border'], log_container_rect, 1)

        # Create a surface for the log messages
        log_surface = pygame.Surface((log_container_rect.width - 10, log_container_rect.height - 10))
        log_surface.fill(self.log_bg_color)

        # Reverse the order of messages to display the latest at the bottom
        visible_messages = list(reversed(self.log_messages[-self.max_log_messages:]))
        
        for i, message in enumerate(visible_messages):
            text_surface, _ = self.font.render(message, self.colors['text'])
            y_position = log_surface.get_height() - (i + 1) * 20 - 5  # Position from bottom
            log_surface.blit(text_surface, (5, y_position))

        # Blit the log surface onto the screen
        self.interface.screen.blit(log_surface, (log_container_rect.left + 5, log_container_rect.top + 5))

    def draw_bottom_panel(self):
        panel_rect = self._get_bottom_panel_rect()
        input_box = self._get_input_box_rect(panel_rect)
        
        pygame.draw.rect(self.interface.screen, self.colors['input_bg'], input_box)
        pygame.draw.rect(self.interface.screen, self.colors['input_border'], input_box, self.BORDER_WIDTH)
        
        self._draw_input_text(input_box)
        self._draw_cursor(input_box)
        self._draw_instructions(panel_rect)

    def _get_bottom_panel_rect(self):
        panel_height = self.INPUT_HEIGHT + self.INSTRUCTION_HEIGHT + 2 * self.PADDING
        return pygame.Rect(0, self.interface.height - panel_height,
                           self.interface.width - self.interface.minimap_panel_width, panel_height)

    def _get_input_box_rect(self, panel_rect):
        return pygame.Rect(self.PADDING, panel_rect.top + self.PADDING,
                           panel_rect.width - 2 * self.PADDING, self.INPUT_HEIGHT)

    def _draw_input_text(self, input_box):
        text_surface, _ = self.font.render(self.interface.direction_text, self.colors['text'])
        self.interface.screen.blit(text_surface, (input_box.x + 10, input_box.y + self.TEXT_VERTICAL_OFFSET))

    def _draw_cursor(self, input_box):
        if self.interface.text_input_active:
            self._update_cursor_state()
            if self.cursor_visible:
                text_surface, _ = self.font.render(self.interface.direction_text, self.colors['text'])
                cursor_x = input_box.x + 10 + text_surface.get_width()
                cursor_y = input_box.y + self.TEXT_VERTICAL_OFFSET
                pygame.draw.line(self.interface.screen, self.colors['text'], 
                                 (cursor_x, cursor_y), (cursor_x, cursor_y + self.CURSOR_HEIGHT), self.CURSOR_WIDTH)

    def _update_cursor_state(self):
        self.cursor_timer += self.interface.clock.get_time()
        if self.cursor_timer >= self.CURSOR_BLINK_TIME:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def _draw_instructions(self, panel_rect):
        instructions = "TAB: Toggle input | WASD: Move | R: Reset | ENTER: Set direction"
        instruction_surface, _ = self.font.render(instructions, self.colors['text'])
        instruction_rect = instruction_surface.get_rect(center=(panel_rect.centerx, 
                                                                panel_rect.bottom - self.INSTRUCTION_HEIGHT // 2))
        self.interface.screen.blit(instruction_surface, instruction_rect)

    @staticmethod
    def pil_image_to_pygame_surface(pil_image):
        return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()

    def add_log_message(self, message):
        self.log_messages.append(message)
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop(0)