import math
import pygame
import sys

# -------------------------------------------------------------
# Cannonball Simulation (faithful port of Matthias Müller's JS)
# -------------------------------------------------------------

pygame.init()

# Window setup
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Cannonball")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

# Simulation space setup
sim_min_width = 20.0
c_scale = min(screen_width, screen_height) / sim_min_width
sim_width = screen_width / c_scale
sim_height = screen_height / c_scale

# Coordinate transforms
def cX(pos):
    return int(pos["x"] * c_scale)

def cY(pos):
    # Y-axis is inverted in pygame vs canvas
    return int(screen_height - pos["y"] * c_scale)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def triangle_orientation(points):
    area2 = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area2 += x1 * y2 - x2 * y1
    if area2 >= 0.0:
        return 1.0
    return -1.0

def point_in_triangle(point, points):
    (x, y) = point
    (x1, y1), (x2, y2), (x3, y3) = points
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if denom == 0.0:
        return False
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1.0 - a - b
    eps = -1e-6
    return a >= eps and b >= eps and c >= eps

def edge_outward_normal(p1, p2, orientation):
    ex = p2[0] - p1[0]
    ey = p2[1] - p1[1]
    if orientation >= 0.0:
        nx, ny = ey, -ex
    else:
        nx, ny = -ey, ex
    length = math.hypot(nx, ny)
    if length == 0.0:
        return 0.0, 0.0
    return nx / length, ny / length

# Physics constants
gravity = {"x": 0.0, "y": -10.0}
time_step = 1.0 / 60.0
boost_accel = 15.0
air_density = 1.2
drag_coefficient = 0.3
house_restitution = 0.6
house_tangent_damping = 0.85

# Ball setup
balls = [
    {
        "radius": 0.2,
        "pos": {"x": 0.2, "y": 0.2},
        "vel": {"x": 10.0, "y": 20.0},
        "color": (255, 0, 0),
        "mass": 1.0,
    },
    {
        "radius": 0.25,
        "pos": {"x": 0.2, "y": 0.5},
        "vel": {"x": 8.0, "y": 18.0},
        "color": (0, 128, 255),
        "mass": 1.2,
    },
    {
        "radius": 0.18,
        "pos": {"x": 0.2, "y": 0.8},
        "vel": {"x": 12.0, "y": 20.0},
        "color": (0, 180, 0),
        "mass": 0.9,
    },
]

house = {
    "base": {
        "x": 20.5,
        "width": 3.0,
        "height": 3.6,
    },
    "roof_height": 2.4,
    "base_color": (168, 126, 84),
    "roof_color": (196, 66, 54),
    "outline_color": (60, 40, 30),
    "restitution": house_restitution,
    "tangent_damping": house_tangent_damping,
}

floating_triangle = {
    "points": [
        (15.0, 6.5),
        (18.5, 6.2),
        (16.5, 9.5),
    ],
    "fill_color": (120, 180, 210),
    "outline_color": (40, 80, 110),
    "restitution": 0.5,
    "tangent_damping": 0.9,
}

button_width, button_height = 120, 30
button_margin = 10
buttons = []
for idx, _ in enumerate(balls):
    rect = pygame.Rect(
        button_margin + idx * (button_width + button_margin),
        screen_height - button_height - button_margin,
        button_width,
        button_height,
    )
    buttons.append(
        {
            "ball_index": idx,
            "rect": rect,
            "label": f"Boost {idx + 1}",
            "is_pressed": False,
        }
    )

def house_vertices():
    base = house["base"]
    x_left = base["x"]
    x_right = x_left + base["width"]
    y_top = base["height"]
    apex_x = x_left + base["width"] * 0.5
    apex_y = y_top + house["roof_height"]
    return {
        "left_corner": (x_left, y_top),
        "right_corner": (x_right, y_top),
        "apex": (apex_x, apex_y),
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
    }

def draw_house():
    verts = house_vertices()
    base = house["base"]
    rect_left = cX({"x": verts["x_left"]})
    rect_top = cY({"y": verts["y_top"]})
    rect_width = int(base["width"] * c_scale)
    rect_height = int(base["height"] * c_scale)
    base_rect = pygame.Rect(rect_left, rect_top, rect_width, rect_height)
    pygame.draw.rect(screen, house["base_color"], base_rect)

    roof_points = [
        (cX({"x": verts["left_corner"][0]}), cY({"y": verts["left_corner"][1]})),
        (cX({"x": verts["right_corner"][0]}), cY({"y": verts["right_corner"][1]})),
        (cX({"x": verts["apex"][0]}), cY({"y": verts["apex"][1]})),
    ]
    pygame.draw.polygon(screen, house["roof_color"], roof_points)
    pygame.draw.rect(screen, house["outline_color"], base_rect, 2)
    pygame.draw.lines(screen, house["outline_color"], True, roof_points, 2)

def draw_floating_triangle():
    pts_pixels = [
        (cX({"x": x}), cY({"y": y}))
        for (x, y) in floating_triangle["points"]
    ]
    pygame.draw.polygon(screen, floating_triangle["fill_color"], pts_pixels)
    pygame.draw.polygon(screen, floating_triangle["outline_color"], pts_pixels, 2)

def resolve_circle_segment(ball, p1, p2, restitution, tangent_damping):
    r = ball["radius"]
    px, py = ball["pos"]["x"], ball["pos"]["y"]
    sx = p2[0] - p1[0]
    sy = p2[1] - p1[1]
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq == 0.0:
        return

    t = ((px - p1[0]) * sx + (py - p1[1]) * sy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    closest_x = p1[0] + t * sx
    closest_y = p1[1] + t * sy
    dx = px - closest_x
    dy = py - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= r * r:
        return

    if dist_sq == 0.0:
        normal_x = -sy
        normal_y = sx
        normal_len = math.hypot(normal_x, normal_y)
        if normal_len == 0.0:
            return
        normal_x /= normal_len
        normal_y /= normal_len
        penetration = r
    else:
        dist = math.sqrt(dist_sq)
        normal_x = dx / dist
        normal_y = dy / dist
        penetration = r - dist

    ball["pos"]["x"] += normal_x * penetration
    ball["pos"]["y"] += normal_y * penetration

    vel_dot_normal = ball["vel"]["x"] * normal_x + ball["vel"]["y"] * normal_y
    if vel_dot_normal < 0.0:
        ball["vel"]["x"] -= (1 + restitution) * vel_dot_normal * normal_x
        ball["vel"]["y"] -= (1 + restitution) * vel_dot_normal * normal_y

        new_vn = ball["vel"]["x"] * normal_x + ball["vel"]["y"] * normal_y
        vt_x = ball["vel"]["x"] - new_vn * normal_x
        vt_y = ball["vel"]["y"] - new_vn * normal_y
        ball["vel"]["x"] = new_vn * normal_x + vt_x * tangent_damping
        ball["vel"]["y"] = new_vn * normal_y + vt_y * tangent_damping

def handle_house_collision(ball):
    verts = house_vertices()
    base = house["base"]
    x_left = verts["x_left"]
    x_right = verts["x_right"]
    y_top = verts["y_top"]
    y_bottom = 0.0
    r = ball["radius"]
    restitution = house["restitution"]
    tangent_damping = house["tangent_damping"]
    px = ball["pos"]["x"]
    py = ball["pos"]["y"]

    expanded_left = x_left - r
    expanded_right = x_right + r
    expanded_bottom = y_bottom - r
    expanded_top = y_top + r

    if (expanded_left <= px <= expanded_right and
            expanded_bottom <= py <= expanded_top):
        overlaps = {
            "left": px - (x_left - r),
            "right": (x_right + r) - px,
            "bottom": py - (y_bottom - r),
            "top": (y_top + r) - py,
        }
        hit_side = min(overlaps, key=overlaps.get)
        penetration = overlaps[hit_side]
        if penetration > 0.0:
            if hit_side == "top":
                ball["pos"]["y"] = y_top + r
                if ball["vel"]["y"] < 0.0:
                    ball["vel"]["y"] = -ball["vel"]["y"] * restitution
                    ball["vel"]["x"] *= tangent_damping
            elif hit_side == "left":
                ball["pos"]["x"] = x_left - r
                if ball["vel"]["x"] > 0.0:
                    ball["vel"]["x"] = -ball["vel"]["x"] * restitution
                    ball["vel"]["y"] *= tangent_damping
            elif hit_side == "right":
                ball["pos"]["x"] = x_right + r
                if ball["vel"]["x"] < 0.0:
                    ball["vel"]["x"] = -ball["vel"]["x"] * restitution
                    ball["vel"]["y"] *= tangent_damping
            elif hit_side == "bottom":
                ball["pos"]["y"] = y_bottom - r
                if ball["vel"]["y"] > 0.0:
                    ball["vel"]["y"] = -ball["vel"]["y"] * restitution
                    ball["vel"]["x"] *= tangent_damping

    apex = verts["apex"]
    left_corner = verts["left_corner"]
    right_corner = verts["right_corner"]
    resolve_circle_segment(ball, apex, left_corner, restitution, tangent_damping)
    resolve_circle_segment(ball, apex, right_corner, restitution, tangent_damping)

def handle_floating_triangle_collision(ball):
    points = floating_triangle["points"]
    restitution = floating_triangle["restitution"]
    tangent_damping = floating_triangle["tangent_damping"]
    orientation = triangle_orientation(points)

    for edge_idx in range(len(points)):
        p1 = points[edge_idx]
        p2 = points[(edge_idx + 1) % len(points)]
        resolve_circle_segment(ball, p1, p2, restitution, tangent_damping)

    if not point_in_triangle((ball["pos"]["x"], ball["pos"]["y"]), points):
        return

    worst_dist = None
    best_normal = (0.0, 0.0)
    for edge_idx in range(len(points)):
        p1 = points[edge_idx]
        p2 = points[(edge_idx + 1) % len(points)]
        normal = edge_outward_normal(p1, p2, orientation)
        if normal == (0.0, 0.0):
            continue
        dist = ((ball["pos"]["x"] - p1[0]) * normal[0] +
                (ball["pos"]["y"] - p1[1]) * normal[1])
        if worst_dist is None or dist < worst_dist:
            worst_dist = dist
            best_normal = normal

    if worst_dist is None:
        return

    shift = ball["radius"] - worst_dist
    if shift <= 0.0:
        return

    ball["pos"]["x"] += best_normal[0] * shift
    ball["pos"]["y"] += best_normal[1] * shift

    vn = ball["vel"]["x"] * best_normal[0] + ball["vel"]["y"] * best_normal[1]
    if vn < 0.0:
        ball["vel"]["x"] -= (1 + restitution) * vn * best_normal[0]
        ball["vel"]["y"] -= (1 + restitution) * vn * best_normal[1]
        new_vn = ball["vel"]["x"] * best_normal[0] + ball["vel"]["y"] * best_normal[1]
        vt_x = ball["vel"]["x"] - new_vn * best_normal[0]
        vt_y = ball["vel"]["y"] - new_vn * best_normal[1]
        ball["vel"]["x"] = new_vn * best_normal[0] + vt_x * tangent_damping
        ball["vel"]["y"] = new_vn * best_normal[1] + vt_y * tangent_damping

# Draw the ball
def draw():
    screen.fill((255, 255, 255))
    draw_house()
    draw_floating_triangle()
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
        fill_color = (200, 200, 200) if not button["is_pressed"] else (150, 220, 150)
        pygame.draw.rect(screen, fill_color, button["rect"])
        pygame.draw.rect(screen, (0, 0, 0), button["rect"], 2)
        label_surface = font.render(button["label"], True, (0, 0, 0))
        label_rect = label_surface.get_rect(center=button["rect"].center)
        screen.blit(label_surface, label_rect)

# Simulate one time step
def simulate():
    for idx, ball in enumerate(balls):
        # Integrate velocity under constant gravity
        ball["vel"]["x"] += gravity["x"] * time_step
        ball["vel"]["y"] += gravity["y"] * time_step

        if buttons[idx]["is_pressed"]:
            ball["vel"]["y"] += boost_accel * time_step

        vx = ball["vel"]["x"]
        vy = ball["vel"]["y"]
        speed = math.hypot(vx, vy)
        if speed > 0.0:
            area = math.pi * ball["radius"] ** 2
            mass = ball.get("mass", 1.0)
            drag_factor = 0.5 * air_density * drag_coefficient * area / mass
            ball["vel"]["x"] -= drag_factor * vx * speed * time_step
            ball["vel"]["y"] -= drag_factor * vy * speed * time_step

        # Integrate position
        ball["pos"]["x"] += ball["vel"]["x"] * time_step
        ball["pos"]["y"] += ball["vel"]["y"] * time_step

        # Collisions with walls
        if ball["pos"]["x"] < 0.0:
            ball["pos"]["x"] = 0.0
            ball["vel"]["x"] = -ball["vel"]["x"]

        if ball["pos"]["x"] > sim_width:
            ball["pos"]["x"] = sim_width
            ball["vel"]["x"] = -ball["vel"]["x"]

        if ball["pos"]["y"] < 0.0:
            ball["pos"]["y"] = 0.0
            ball["vel"]["y"] = -ball["vel"]["y"]

        handle_house_collision(ball)
        handle_floating_triangle_collision(ball)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in buttons:
                if button["rect"].collidepoint(event.pos):
                    button["is_pressed"] = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for button in buttons:
                button["is_pressed"] = False

    simulate()
    draw()

    clock.tick(60)
