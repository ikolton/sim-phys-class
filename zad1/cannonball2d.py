import math
import pygame
import sys

pygame.init()

# Okno
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Cannonball – Billiard Style")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

# Skala symulacji
sim_min_width = 20.0
c_scale = min(screen_width, screen_height) / sim_min_width
sim_width = screen_width / c_scale
sim_height = screen_height / c_scale

def cX(pos): return int(pos["x"] * c_scale)
def cY(pos): return int(screen_height - pos["y"] * c_scale)

# Stałe fizyczne
gravity = {"x": 0.0, "y": -9.8}
time_step = 1.0 / 60.0
restitution = 1.0
air_density = 1.1
drag_coefficient = 0.1
bounce_strength = 15.0   # prędkość "odbicia" w górę

# Kule
balls = []
num_balls = 10
for i in range(num_balls):
    radius = 0.45 + 0.15 * math.sin(i)
    mass = math.pi * radius * radius
    balls.append({
        "radius": radius,
        "mass": mass,
        "pos": {"x": 2.0 + i * 1.2, "y": 5.0 + 0.5 * (i % 2)},
        "vel": {"x": (i % 3) * 10.0, "y": (2 - i % 3) * 10.0},
        "color": (255 - i * 20, 50 + i * 20, 150 + i * 10),
    })

# Przyciski
button_width, button_height = 120, 30
button_margin = 10
max_buttons = 3  
buttons = []
for idx, _ in enumerate(balls[:max_buttons]):  # tylko pierwsze 3 kule
    rect = pygame.Rect(
        button_margin + idx * (button_width + button_margin),
        screen_height - button_height - button_margin,
        button_width,
        button_height,
    )
    buttons.append({
        "ball_index": idx,
        "rect": rect,
        "label": f"Bounce {idx + 1}",
    })

# --- Kolizje kul i ścian --------------------------------------

def handle_wall_collision(ball):
    """Odbicie od ścian świata."""
    r = ball["radius"]
    if ball["pos"]["x"] < r:
        ball["pos"]["x"] = r
        ball["vel"]["x"] = -ball["vel"]["x"]
    if ball["pos"]["x"] > sim_width - r:
        ball["pos"]["x"] = sim_width - r
        ball["vel"]["x"] = -ball["vel"]["x"]
    if ball["pos"]["y"] < r:
        ball["pos"]["y"] = r
        ball["vel"]["y"] = -ball["vel"]["y"]
    if ball["pos"]["y"] > sim_height - r:
        ball["pos"]["y"] = sim_height - r
        ball["vel"]["y"] = -ball["vel"]["y"]

def handle_ball_collision(ball1, ball2):
    """Kolizje między kulami (elastic)."""
    dx = ball2["pos"]["x"] - ball1["pos"]["x"]
    dy = ball2["pos"]["y"] - ball1["pos"]["y"]
    dist = math.hypot(dx, dy)
    r_sum = ball1["radius"] + ball2["radius"]
    if dist == 0.0 or dist > r_sum:
        return

    # Normalizacja kierunku
    nx = dx / dist
    ny = dy / dist

    # Korekcja pozycji (penetracja)
    corr = (r_sum - dist) / 2.0
    ball1["pos"]["x"] -= nx * corr
    ball1["pos"]["y"] -= ny * corr
    ball2["pos"]["x"] += nx * corr
    ball2["pos"]["y"] += ny * corr

    # Składowe prędkości wzdłuż normalnej
    v1 = ball1["vel"]["x"] * nx + ball1["vel"]["y"] * ny
    v2 = ball2["vel"]["x"] * nx + ball2["vel"]["y"] * ny

    m1 = ball1["mass"]
    m2 = ball2["mass"]

    new_v1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    new_v2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

    ball1["vel"]["x"] += (new_v1 - v1) * nx
    ball1["vel"]["y"] += (new_v1 - v1) * ny
    ball2["vel"]["x"] += (new_v2 - v2) * nx
    ball2["vel"]["y"] += (new_v2 - v2) * ny

# --- Rysowanie ------------------------------------------------

def draw():
    screen.fill((255, 255, 255))
    for ball in balls:
        pygame.draw.circle(
            screen,
            ball["color"],
            (cX(ball["pos"]), cY(ball["pos"])),
            int(c_scale * ball["radius"])
        )
    draw_buttons()
    pygame.display.flip()

def draw_buttons():
    for button in buttons:
        pygame.draw.rect(screen, (200, 200, 200), button["rect"])
        pygame.draw.rect(screen, (0, 0, 0), button["rect"], 2)
        label_surface = font.render(button["label"], True, (0, 0, 0))
        label_rect = label_surface.get_rect(center=button["rect"].center)
        screen.blit(label_surface, label_rect)

# --- Symulacja ------------------------------------------------

def simulate():
    for ball in balls:
        # --- Grawitacja
        ball["vel"]["x"] += gravity["x"] * time_step
        ball["vel"]["y"] += gravity["y"] * time_step

        # --- Opor powietrza
        vx, vy = ball["vel"]["x"], ball["vel"]["y"]
        speed = math.hypot(vx, vy)
        if speed > 0.0:
            area = math.pi * ball["radius"] ** 2
            mass = ball["mass"]
            drag_factor = 0.5 * air_density * drag_coefficient * area / mass
            ball["vel"]["x"] -= drag_factor * vx * speed * time_step
            ball["vel"]["y"] -= drag_factor * vy * speed * time_step

        # --- Ruch
        ball["pos"]["x"] += ball["vel"]["x"] * time_step
        ball["pos"]["y"] += ball["vel"]["y"] * time_step

        handle_wall_collision(ball)

    # --- Kolizje między kulami
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            handle_ball_collision(balls[i], balls[j])

# --- Pętla główna ---------------------------------------------

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in buttons:
                if button["rect"].collidepoint(event.pos):
                    # 👇 natychmiastowe "odbicie w górę"
                    ball = balls[button["ball_index"]]
                    ball["vel"]["y"] = abs(bounce_strength)
                    # random x boost with modulo and random num generator
                    ball["vel"]["x"] += (math.sin(pygame.time.get_ticks() % 1000) * 10.0)

    simulate()
    draw()
    clock.tick(60)
