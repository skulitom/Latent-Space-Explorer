import pygame

class Minimap:
    def __init__(self, interface):
        self.interface = interface
        self.setup_minimap()

    def setup_minimap(self):
        self.minimap_size = 180
        self.minimap_surface = pygame.Surface((self.minimap_size, self.minimap_size))
        self.explored_positions = [(0, 0)]  # Start at center
        self.minimap_scale = 10  # Pixels per step
        self.minimap_bg = (40, 40, 40)
        self.minimap_border = (60, 60, 60)
        self.minimap_path = (100, 100, 100)
        self.minimap_current = (0, 255, 0)

    def draw(self, screen):
        self.minimap_surface.fill(self.minimap_bg)
        
        center = self.minimap_size // 2
        for i, pos in enumerate(self.explored_positions):
            x = center + pos[0] * self.minimap_scale
            y = center + pos[1] * self.minimap_scale
            if i == 0:
                pygame.draw.circle(self.minimap_surface, (255, 255, 255), (int(x), int(y)), 4)
            if i > 0:
                prev_pos = self.explored_positions[i-1]
                prev_x = center + prev_pos[0] * self.minimap_scale
                prev_y = center + prev_pos[1] * self.minimap_scale
                pygame.draw.line(self.minimap_surface, self.minimap_path, (prev_x, prev_y), (x, y), 2)
        
        current_pos = self.explored_positions[-1]
        current_x = center + current_pos[0] * self.minimap_scale
        current_y = center + current_pos[1] * self.minimap_scale
        pygame.draw.circle(self.minimap_surface, self.minimap_current, (int(current_x), int(current_y)), 4)

        pygame.draw.rect(self.minimap_surface, self.minimap_border, (0, 0, self.minimap_size, self.minimap_size), 2)

        minimap_rect = pygame.Rect(self.interface.width + 10, 40, self.minimap_size, self.minimap_size)
        screen.blit(self.minimap_surface, minimap_rect)

        # Draw minimap title
        title_font = pygame.font.SysFont("Arial", 20)
        title_surface = title_font.render("Latent Space Map", True, (220, 220, 220))
        title_rect = title_surface.get_rect(center=(self.interface.width + self.interface.minimap_panel_width // 2, 20))
        screen.blit(title_surface, title_rect)

    def update_position(self, move_direction):
        last_pos = self.explored_positions[-1]
        if move_direction == 'forward':
            new_pos = (last_pos[0], last_pos[1] - 1)
        elif move_direction == 'backward':
            new_pos = (last_pos[0], last_pos[1] + 1)
        elif move_direction == 'left':
            new_pos = (last_pos[0] - 1, last_pos[1])
        elif move_direction == 'right':
            new_pos = (last_pos[0] + 1, last_pos[1])
        self.explored_positions.append(new_pos)

    def reset_position(self):
        self.explored_positions = [(0, 0)]