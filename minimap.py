import pygame

class Minimap:
    def __init__(self, interface):
        self.interface = interface
        self.setup_minimap()

    def setup_minimap(self):
        self.minimap_size = min(220, self.interface.minimap_panel_width - 20)  # Increased size
        self.minimap_surface = pygame.Surface((self.minimap_size, self.minimap_size))
        self.explored_positions = [(0, 0)]  # Start at center
        self.current_x, self.current_y = 0, 0
        self.minimap_scale = 8  # Reduced scale to show more area
        self.minimap_bg = (40, 40, 40)
        self.minimap_border = (60, 60, 60)
        self.minimap_path = (100, 100, 100)
        self.minimap_current = (0, 255, 0)

    def draw(self, screen, rect):
        self.minimap_surface.fill(self.minimap_bg)
        
        center_x, center_y = self.minimap_size // 2, self.minimap_size // 2
        
        # Draw explored positions
        for pos in self.explored_positions:
            x = center_x + pos[0] * self.minimap_scale
            y = center_y + pos[1] * self.minimap_scale
            pygame.draw.circle(self.minimap_surface, self.minimap_path, (x, y), 2)
        
        # Draw current position
        current_x = center_x + self.current_x * self.minimap_scale
        current_y = center_y + self.current_y * self.minimap_scale
        pygame.draw.circle(self.minimap_surface, self.minimap_current, (current_x, current_y), 4)
        
        # Draw border
        pygame.draw.rect(self.minimap_surface, self.minimap_border, (0, 0, self.minimap_size, self.minimap_size), 2)
        
        # Blit minimap surface onto the screen
        screen.blit(self.minimap_surface, (rect.left, rect.top))

    def update_position(self, move_direction):
        if move_direction == 'forward':
            self.current_y -= 1
        elif move_direction == 'backward':
            self.current_y += 1
        elif move_direction == 'left':
            self.current_x -= 1
        elif move_direction == 'right':
            self.current_x += 1
        
        new_pos = (self.current_x, self.current_y)
        if new_pos not in self.explored_positions:
            self.explored_positions.append(new_pos)

    def reset_position(self):
        self.explored_positions = [(0, 0)]
        self.current_x, self.current_y = 0, 0