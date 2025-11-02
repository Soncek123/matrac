import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Circle with Mouse")

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)

# Circle settings
circle_pos = [WIDTH // 2, HEIGHT // 2]
circle_radius = 50
dragging = False

# Main loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            dx = mouse_x - circle_pos[0]
            dy = mouse_y - circle_pos[1]
            # Check if mouse is inside the circle
            if dx * dx + dy * dy <= circle_radius * circle_radius:
                dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                circle_pos = list(event.pos)

    # Draw
    screen.fill(WHITE)
    pygame.draw.circle(screen, BLUE, circle_pos, circle_radius)
    pygame.display.flip()

    clock.tick(60)
