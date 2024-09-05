import pygame
import pygame.freetype
import pygame.gfxdraw
from PIL import Image
from latent_explorer import LatentSpaceExplorer
import asyncio

async def create_latent_walk_interface(cfg_path, cfg, flux_model):
    explorer = LatentSpaceExplorer(flux_model, cfg)

    pygame.init()
    screen = pygame.display.set_mode((cfg["width"], cfg["height"] + 100), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    pygame.freetype.init()

    prompt_text = cfg['prompt']
    
    print("Generating initial image...")
    _, current_image = await flux_model.run_flux_inference_async(prompt_text)
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
    
    # Update UI colors for a more modern look
    ui_colors = cfg.get('ui_colors', {})
    bg_color = ui_colors.get('background', (18, 18, 18))
    input_bg = ui_colors.get('text_input_bg', (30, 30, 30))
    input_border = ui_colors.get('text_input_border', (45, 45, 45))
    text_color = ui_colors.get('text_color', (230, 230, 230))
    accent_color = ui_colors.get('accent_color', (0, 120, 215))
    instruction_color = ui_colors.get('instruction_color', (180, 180, 180))

    # Set up window caption and icon
    pygame.display.set_caption("Latent Space Explorer")
    icon = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(icon, accent_color, (16, 16), 14)
    pygame.draw.circle(icon, bg_color, (16, 16), 10)
    pygame.display.set_icon(icon)

    # Load or create a modern font
    try:
        font = pygame.freetype.Font("path/to/modern/font.ttf", 18)
    except:
        font = pygame.freetype.SysFont("Arial", 18)

    def draw_ui():
        screen_width, screen_height = screen.get_size()
        image_height = screen_height - 100

        screen.fill(bg_color)

        # Draw the image without white lines
        scaled_image = pygame.transform.scale(pygame_image, (screen_width - 40, image_height - 20))
        image_rect = scaled_image.get_rect(center=(screen_width // 2, (image_height - 20) // 2 + 10))
        screen.blit(scaled_image, image_rect)

        # Draw a border around the image
        pygame.draw.rect(screen, input_border, image_rect, 1)

        # Draw a modern, minimal input box
        input_box = pygame.Rect(20, image_height + 10, screen_width - 40, 40)
        pygame.draw.rect(screen, input_bg, input_box)
        pygame.draw.rect(screen, input_border, input_box, 1)

        text_surface, _ = font.render(direction_text, text_color)
        screen.blit(text_surface, (input_box.x + 15, input_box.y + 10))

        if text_input_active and text_cursor_visible:
            cursor_pos = font.get_rect(direction_text[:len(direction_text)])
            pygame.draw.line(screen, text_color,
                             (input_box.x + 15 + cursor_pos.width, input_box.y + 8),
                             (input_box.x + 15 + cursor_pos.width, input_box.y + 32), 2)

        # Draw modern, minimal instructions
        instructions = [
            "TAB: Toggle input | WASD: Move | R: Reset | ENTER: Set direction"
        ]
        instruction_surface, _ = font.render(instructions[0], instruction_color)
        instruction_rect = instruction_surface.get_rect(center=(screen_width // 2, screen_height - 25))
        screen.blit(instruction_surface, instruction_rect)

        if current_direction and pygame.time.get_ticks() - direction_change_time < direction_display_duration:
            direction_text_surface, _ = font.render(f"Moving: {current_direction}", accent_color)
            text_rect = direction_text_surface.get_rect(center=(screen_width // 2, image_height + 70))
            screen.blit(direction_text_surface, text_rect)

    async def update_image(new_prompt):
        new_image = await explorer.generate_image(new_prompt)
        return pil_image_to_pygame_surface(new_image)

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
                        await explorer.update_latents(prompt_text, direction_text, 'forward', 0)
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
                _, reset_prompt = await explorer.reset_position()
                pygame_image = await update_image(reset_prompt)
                print("Reset completed")

            if move_direction and direction_text:
                try:
                    print(f"Attempting to move: {move_direction}")
                    _, new_prompt = await explorer.update_latents(prompt_text, direction_text, move_direction, cfg['step_size'])
                    pygame_image = await update_image(new_prompt)
                    print(f"Move completed successfully. New prompt: {new_prompt}")
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

        await asyncio.sleep(0)  # Allow other async tasks to run

    pygame.quit()

def pil_image_to_pygame_surface(pil_image: Image.Image) -> pygame.Surface:
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode).convert()