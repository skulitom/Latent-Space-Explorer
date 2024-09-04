import pygame
import pygame.freetype
from PIL import Image
from latent_explorer import LatentSpaceExplorer

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
        if corner_radius < 0:
            raise ValueError(f"Corner radius ({corner_radius}) must be >= 0")

        rect_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(rect_surface, color, (0, 0, *rect.size), border_radius=corner_radius)
        surface.blit(rect_surface, rect)

    def create_gradient(width, height, start_color, end_color):
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

        scaled_image = pygame.transform.scale(pygame_image, (screen_width, image_height))
        screen.blit(scaled_image, (0, 0))

        gradient = create_gradient(screen_width, 120, bg_start, bg_end)
        screen.blit(gradient, (0, image_height))

        input_box = pygame.Rect(20, image_height + 20, screen_width - 40, 40)
        draw_rounded_rect(screen, input_box, input_bg, 10)
        
        pygame.draw.rect(screen, input_border, input_box, 2, border_radius=10)

        text_surface, _ = font.render(direction_text, text_color)
        screen.blit(text_surface, (input_box.x + 10, input_box.y + 10))

        if text_input_active and text_cursor_visible:
            cursor_pos = font.get_rect(direction_text[:len(direction_text)])
            pygame.draw.line(screen, text_color,
                             (input_box.x + 10 + cursor_pos.width, input_box.y + 10),
                             (input_box.x + 10 + cursor_pos.width, input_box.y + 30), 2)

        instructions = [
            "TAB: Toggle text input | WASD: Move | R: Reset",
            "Enter: Submit direction"
        ]
        for i, instruction in enumerate(instructions):
            text_surface, _ = font.render(instruction, instruction_color)
            screen.blit(text_surface, (20, image_height + 70 + i * 25))

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
                    latent_noise, new_prompt = explorer.update_latents(latent_noise, prompt_text, direction_text, move_direction, cfg['step_size'])
                    _, new_image = flux_model.run_flux_inference(new_prompt)
                    pygame_image = pil_image_to_pygame_surface(new_image)
                    print("Move completed successfully")
                    current_direction = move_direction
                    direction_change_time = pygame.time.get_ticks()
                except Exception as e:
                    print(f"Error during image update: {e}")
                    import traceback
                    traceback.print_exc()

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