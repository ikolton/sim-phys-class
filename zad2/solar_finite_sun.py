import math
import random
import pygame


SIM_MIN_WIDTH = 2.0
WINDOW_SIZE = (960, 720)
BACKGROUND_COLOR = (255, 255, 255)
WIRE_COLOR = (180, 180, 180)
BEAD_COLOR = (220, 30, 30)
SUN_COLOR = (255, 180, 0)
SUN_MASS_OPTIONS = [250.0, 500.0, 1000.0, 5000.0]


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


class Particle:
    __slots__ = ("mass", "inv_mass", "pos", "prev_pos", "vel")

    def __init__(self, mass, pos):
        self.mass = mass
        self.inv_mass = 0.0 if mass <= 0.0 else 1.0 / mass
        self.pos = pos.clone()
        self.prev_pos = pos.clone()
        self.vel = Vector2()

    def set_mass(self, mass):
        self.mass = mass
        self.inv_mass = 0.0 if mass <= 0.0 else 1.0 / mass

    def start_step(self, dt, gravity=None):
        if gravity is not None and self.inv_mass > 0.0:
            self.vel.add(gravity, dt)
        self.prev_pos.set(self.pos)
        self.pos.add(self.vel, dt)

    def end_step(self, dt):
        self.vel.subtract_vectors(self.pos, self.prev_pos)
        self.vel.scale(1.0 / dt)


class Bead(Particle):
    __slots__ = Particle.__slots__ + ("radius",)

    def __init__(self, radius, mass, pos):
        super().__init__(mass, pos)
        self.radius = radius

    def keep_on_wire(self, sun, radius):
        direction = Vector2()
        direction.subtract_vectors(self.pos, sun.pos)
        length = direction.length()
        if length == 0.0:
            # Small random nudge avoids division by zero
            direction.x = random.uniform(-1.0, 1.0)
            direction.y = random.uniform(-1.0, 1.0)
            length = direction.length()
            if length == 0.0:
                return
        direction.scale(1.0 / length)
        constraint = length - radius
        if abs(constraint) < 1e-6:
            return
        inv_mass_sum = self.inv_mass + sun.inv_mass
        if inv_mass_sum == 0.0:
            return
        correction = constraint / inv_mass_sum
        self.pos.add(direction, -correction * self.inv_mass)
        sun.pos.add(direction, correction * sun.inv_mass)


class Sun(Particle):
    __slots__ = Particle.__slots__ + ("radius",)

    def __init__(self, mass, pos, radius=0.12):
        super().__init__(mass, pos)
        self.radius = radius


class PhysicsScene:
    __slots__ = (
        "gravity",
        "dt",
        "num_steps",
        "wire_radius",
        "beads",
        "sun",
        "sun_mass_options",
        "sun_mass_index",
    )

    def __init__(self):
        self.gravity = Vector2(0.0, -10.0)
        self.dt = 1.0 / 60.0
        self.num_steps = 100
        self.wire_radius = 0.0
        self.beads = []
        self.sun_mass_options = SUN_MASS_OPTIONS
        self.sun_mass_index = min(2, len(self.sun_mass_options) - 1)
        self.sun = None

    @property
    def sun_mass(self):
        return self.sun_mass_options[self.sun_mass_index]


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

    sun_pos = Vector2(canvas.sim_width / 2.0, canvas.sim_height / 2.0)
    scene.sun = Sun(scene.sun_mass, sun_pos)
    scene.wire_radius = SIM_MIN_WIDTH * 0.4

    num_beads = 5
    radius = 0.1
    angle = 0.0
    for _ in range(num_beads):
        mass = math.pi * radius * radius
        pos = Vector2(
            scene.sun.pos.x + scene.wire_radius * math.cos(angle),
            scene.sun.pos.y + scene.wire_radius * math.sin(angle),
        )
        scene.beads.append(Bead(radius, mass, pos))
        angle += math.pi / num_beads
        radius = 0.05 + random.random() * 0.1

    pygame.display.set_caption(
        f"Finite Sun | mass={scene.sun.mass} | beads={len(scene.beads)}"
    )


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
        scene.sun.start_step(step_dt)

        for bead in scene.beads:
            bead.start_step(step_dt, scene.gravity)

        for bead in scene.beads:
            bead.keep_on_wire(scene.sun, scene.wire_radius)

        scene.sun.end_step(step_dt)
        for bead in scene.beads:
            bead.end_step(step_dt)

        bead_count = len(scene.beads)
        for i in range(bead_count):
            for j in range(i):
                handle_bead_bead_collision(scene.beads[i], scene.beads[j])


def draw(screen, scene, canvas):
    screen.fill(BACKGROUND_COLOR)

    center = (
        int(canvas.cx(scene.sun.pos)),
        int(canvas.cy(scene.sun.pos)),
    )
    pygame.draw.circle(
        screen,
        WIRE_COLOR,
        center,
        int(scene.wire_radius * canvas.c_scale),
        width=2,
    )

    pygame.draw.circle(
        screen,
        SUN_COLOR,
        center,
        max(3, int(scene.sun.radius * canvas.c_scale)),
    )

    for bead in scene.beads:
        position = (int(canvas.cx(bead.pos)), int(canvas.cy(bead.pos)))
        pygame.draw.circle(
            screen,
            BEAD_COLOR,
            position,
            max(1, int(bead.radius * canvas.c_scale)),
        )

    pygame.display.flip()


def cycle_sun_mass(scene, direction):
    count = len(scene.sun_mass_options)
    if count == 0:
        return
    scene.sun_mass_index = (scene.sun_mass_index + direction) % count
    print(f"Switched sun mass to {scene.sun_mass:.1f}")


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()

    canvas = CanvasMetrics(*WINDOW_SIZE)
    scene = PhysicsScene()
    setup_scene(scene, canvas)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    setup_scene(scene, canvas)
                elif event.key == pygame.K_LEFT:
                    cycle_sun_mass(scene, -1)
                    setup_scene(scene, canvas)
                elif event.key == pygame.K_RIGHT:
                    cycle_sun_mass(scene, 1)
                    setup_scene(scene, canvas)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    if index < len(scene.sun_mass_options):
                        scene.sun_mass_index = index
                        print(f"Switched sun mass to {scene.sun_mass:.1f}")
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
