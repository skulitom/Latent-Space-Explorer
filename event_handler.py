import pygame

class EventHandler:
    def __init__(self, interface):
        self.interface = interface

    async def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.interface.running = False
            elif event.type == pygame.KEYDOWN:
                await self.handle_keydown(event)

    async def handle_keydown(self, event):
        if event.key == pygame.K_TAB:
            self.interface.text_input_active = not self.interface.text_input_active
        elif self.interface.text_input_active:
            if event.key == pygame.K_RETURN:
                print(f"Direction set to: {self.interface.direction_text}")
                self.interface.text_input_active = False
                self.interface.set_direction()  # This line is now correct
            elif event.key == pygame.K_BACKSPACE:
                self.interface.direction_text = self.interface.direction_text[:-1]
            else:
                self.interface.direction_text += event.unicode
        elif event.key == pygame.K_r:
            print("Resetting to original image")
            self.interface.task_queue.append(self.interface.reset_image())
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
            
            if move_direction and self.interface.direction_text:
                self.interface.task_queue.append(self.interface.move_image(move_direction))