import math
import random
import pygame


SIM_MIN_WIDTH = 2.0
WINDOW_SIZE = (960, 720)
BACKGROUND_COLOR = (255, 255, 255)
WIRE_COLOR = (255, 0, 0)


class Vector2:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def set(self, other):
        self.x = other.x
        self.y = other.y

    def clone(self):
        return Vector2(self.x, self.y)

    def add(self, other, scale=1.0):
        self.x += other.x * scale
        self.y += other.y * scale
        return self

    def add_vectors(self, a, b):
        self.x = a.x + b.x
        self.y = a.y + b.y
        return self

    def subtract(self, other, scale=1.0):
        self.x -= other.x * scale
        self.y -= other.y * scale
        return self

    def subtract_vectors(self, a, b):
        self.x = a.x - b.x
        self.y = a.y - b.y
        return self

    def length(self):
        return math.hypot(self.x, self.y)

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        return self

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def perp(self):
        return Vector2(-self.y, self.x)


class Bead:
    __slots__ = ("radius", "mass", "pos", "prev_pos", "vel")

    def __init__(self, radius, mass, pos):
        self.radius = radius
        self.mass = mass
        self.pos = pos.clone()
        self.prev_pos = pos.clone()
        self.vel = Vector2()

    def start_step(self, dt, gravity):
        self.vel.add(gravity, dt)
        self.prev_pos.set(self.pos)
        self.pos.add(self.vel, dt)

    def keep_on_wire(self, center, radius):
        direction = Vector2()
        direction.subtract_vectors(self.pos, center)
        length = direction.length()
        if length == 0.0:
            return 0.0
        direction.scale(1.0 / length)
        lam = radius - length
        self.pos.add(direction, lam)
        return lam

    def end_step(self, dt):
        self.vel.subtract_vectors(self.pos, self.prev_pos)
        self.vel.scale(1.0 / dt)


class PhysicsScene:
    __slots__ = ("gravity", "dt", "num_steps", "wire_center", "wire_radius", "beads")

    def __init__(self):
        self.gravity = Vector2(0.0, -10.0)
        self.dt = 1.0 / 60.0
        self.num_steps = 100
        self.wire_center = Vector2()
        self.wire_radius = 0.0
        self.beads = []


class CanvasMetrics:
    __slots__ = ("width", "height", "c_scale", "sim_width", "sim_height")

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.update(width, height)

    def update(self, width=None, height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        self.c_scale = min(self.width, self.height) / SIM_MIN_WIDTH
        self.sim_width = self.width / self.c_scale
        self.sim_height = self.height / self.c_scale

    def cx(self, pos):
        return pos.x * self.c_scale

    def cy(self, pos):
        return self.height - pos.y * self.c_scale


def setup_scene(scene, canvas):
    scene.beads = []

    scene.wire_center.x = canvas.sim_width / 2.0
    scene.wire_center.y = canvas.sim_height / 2.0
    scene.wire_radius = SIM_MIN_WIDTH * 0.4

    num_beads = 5
    radius = 0.1
    angle = 0.0
    for _ in range(num_beads):
        mass = math.pi * radius * radius
        pos = Vector2(
            scene.wire_center.x + scene.wire_radius * math.cos(angle),
            scene.wire_center.y + scene.wire_radius * math.sin(angle),
        )
        scene.beads.append(Bead(radius, mass, pos))
        angle += math.pi / num_beads
        radius = 0.05 + random.random() * 0.1


def handle_bead_bead_collision(bead1, bead2):
    restitution = 1.0
    direction = Vector2()
    direction.subtract_vectors(bead2.pos, bead1.pos)
    distance = direction.length()
    if distance == 0.0 or distance > bead1.radius + bead2.radius:
        return

    direction.scale(1.0 / distance)
    correction = (bead1.radius + bead2.radius - distance) / 2.0
    bead1.pos.add(direction, -correction)
    bead2.pos.add(direction, correction)

    v1 = bead1.vel.dot(direction)
    v2 = bead2.vel.dot(direction)

    m1 = bead1.mass
    m2 = bead2.mass
    new_v1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    new_v2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

    bead1.vel.add(direction, new_v1 - v1)
    bead2.vel.add(direction, new_v2 - v2)


def simulate(scene):
    step_dt = scene.dt / scene.num_steps
    for _ in range(scene.num_steps):
        for bead in scene.beads:
            bead.start_step(step_dt, scene.gravity)

        for bead in scene.beads:
            bead.keep_on_wire(scene.wire_center, scene.wire_radius)

        for bead in scene.beads:
            bead.end_step(step_dt)

        bead_count = len(scene.beads)
        for i in range(bead_count):
            for j in range(i):
                handle_bead_bead_collision(scene.beads[i], scene.beads[j])


def draw(screen, scene, canvas):
    screen.fill(BACKGROUND_COLOR)

    center = (
        int(canvas.cx(scene.wire_center)),
        int(canvas.cy(scene.wire_center)),
    )
    pygame.draw.circle(
        screen,
        WIRE_COLOR,
        center,
        int(scene.wire_radius * canvas.c_scale),
        width=2,
    )

    for bead in scene.beads:
        position = (int(canvas.cx(bead.pos)), int(canvas.cy(bead.pos)))
        pygame.draw.circle(
            screen,
            WIRE_COLOR,
            position,
            max(1, int(bead.radius * canvas.c_scale)),
        )

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Constrained Dynamics")
    clock = pygame.time.Clock()

    canvas = CanvasMetrics(*WINDOW_SIZE)
    scene = PhysicsScene()
    setup_scene(scene, canvas)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                setup_scene(scene, canvas)
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                canvas.update(event.w, event.h)
                setup_scene(scene, canvas)

        simulate(scene)
        draw(screen, scene, canvas)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
