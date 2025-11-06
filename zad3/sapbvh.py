import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from direct.gui.DirectGui import DirectButton
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    TextNode,
    Vec3,
)


BOX_COUNT = 5000
WORLD_SIZE = 100.0
MIN_BOX_SIZE = 0.5
MAX_BOX_SIZE = 3.0
MAX_SPEED = 0.5


def expand_bits(value: int) -> int:
    """Bit spreading helper for Morton code calculation."""
    value = (value * 0x00010001) & 0xFF0000FF
    value = (value * 0x00000101) & 0x0F00F00F
    value = (value * 0x00000011) & 0xC30C30C3
    value = (value * 0x00000005) & 0x49249249
    return value


def calculate_morton_code(x: float, y: float, z: float) -> int:
    """Return 30-bit Morton code for a 3D point."""
    def normalize(coord: float) -> float:
        return (coord + WORLD_SIZE / 2.0) / WORLD_SIZE

    nx = max(0.0, min(1.0, normalize(x)))
    ny = max(0.0, min(1.0, normalize(y)))
    nz = max(0.0, min(1.0, normalize(z)))

    ix = min(int(nx * 1023.0), 1023)
    iy = min(int(ny * 1023.0), 1023)
    iz = min(int(nz * 1023.0), 1023)

    xx = expand_bits(ix)
    yy = expand_bits(iy)
    zz = expand_bits(iz)
    return xx | (yy << 1) | (zz << 2)


@dataclass
class AABB:
    minimum: Vec3
    maximum: Vec3


def aabb_intersect(first: AABB, second: AABB) -> bool:
    """Check if two axis-aligned bounding boxes intersect."""
    return (
        first.minimum.x <= second.maximum.x
        and first.maximum.x >= second.minimum.x
        and first.minimum.y <= second.maximum.y
        and first.maximum.y >= second.minimum.y
        and first.minimum.z <= second.maximum.z
        and first.maximum.z >= second.minimum.z
    )


def build_unit_box() -> NodePath:
    """Create a unit cube geometry that can be instanced for every box."""
    format_ = GeomVertexFormat.get_v3n3()
    vertex_data = GeomVertexData("box", format_, Geom.UH_static)
    vertex = GeomVertexWriter(vertex_data, "vertex")
    normal = GeomVertexWriter(vertex_data, "normal")

    faces: List[Tuple[Tuple[float, float, float], ...]] = [
        # Bottom  z = -0.5  normal (0, 0, -1)
        ((-0.5, -0.5, -0.5), (-0.5,  0.5, -0.5), ( 0.5,  0.5, -0.5), ( 0.5, -0.5, -0.5)),
        # Top     z =  0.5  normal (0, 0, 1)
        ((-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5), ( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5)),
        # Left    x = -0.5  normal (-1, 0, 0)
        ((-0.5, -0.5, -0.5), (-0.5, -0.5,  0.5), (-0.5,  0.5,  0.5), (-0.5,  0.5, -0.5)),
        # Right   x =  0.5  normal (1, 0, 0)
        (( 0.5, -0.5, -0.5), ( 0.5,  0.5, -0.5), ( 0.5,  0.5,  0.5), ( 0.5, -0.5,  0.5)),
        # Front   y =  0.5  normal (0, 1, 0)
        ((-0.5,  0.5, -0.5), (-0.5,  0.5,  0.5), ( 0.5,  0.5,  0.5), ( 0.5,  0.5, -0.5)),
        # Back    y = -0.5  normal (0, -1, 0)
        ((-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5), ( 0.5, -0.5,  0.5), (-0.5, -0.5,  0.5)),
    ]

    normals: List[Tuple[float, float, float]] = [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
    ]

    triangles = GeomTriangles(Geom.UH_static)
    index = 0
    for face, face_normal in zip(faces, normals):
        for vertex_pos in face:
            vertex.add_data3(*vertex_pos)
            normal.add_data3(*face_normal)
        triangles.add_vertices(index, index + 1, index + 2)
        triangles.add_vertices(index, index + 2, index + 3)
        index += 4

    geom = Geom(vertex_data)
    geom.add_primitive(triangles)
    node = GeomNode("box")
    node.add_geom(geom)
    node_path = NodePath(node)
    node_path.setShaderAuto()
    return node_path


class Box:
    def __init__(self, box_id: int, parent: NodePath, prototype: NodePath) -> None:
        self.box_id = box_id
        self.width = random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE)
        self.depth = random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE)
        self.height = random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE)

        self.position = Vec3(
            (random.random() - 0.5) * WORLD_SIZE,
            (random.random() - 0.5) * WORLD_SIZE,
            (random.random() - 0.5) * WORLD_SIZE + WORLD_SIZE / 4.0,
        )
        self.velocity = Vec3(
            (random.random() - 0.5) * MAX_SPEED,
            (random.random() - 0.5) * MAX_SPEED,
            (random.random() - 0.5) * MAX_SPEED,
        )

        self.node = prototype.copy_to(parent)
        self.node.setScale(self.width, self.depth, self.height)
        self.node.setPos(self.position)
        self.node.setColor(0.0, 1.0, 0.0, 1.0)
        self.node.setShaderAuto()

        self.is_colliding = False

    def update(self, dt: float) -> None:
        step = dt * 60.0
        self.position += self.velocity * step

        half_width = self.width / 2.0
        half_depth = self.depth / 2.0
        half_height = self.height / 2.0
        limit = WORLD_SIZE / 2.0

        if abs(self.position.x) > limit - half_width:
            self.velocity.x *= -1.0
            self.position.x = math.copysign(limit - half_width, self.position.x)
        if abs(self.position.y) > limit - half_depth:
            self.velocity.y *= -1.0
            self.position.y = math.copysign(limit - half_depth, self.position.y)
        if abs(self.position.z) > limit - half_height:
            self.velocity.z *= -1.0
            self.position.z = math.copysign(limit - half_height, self.position.z)

        self.node.setPos(self.position)

        if self.is_colliding:
            self.node.setColor(0.0, 1.0, 0.0, 1.0)
            self.is_colliding = False

    def mark_collision(self) -> None:
        self.node.setColor(1.0, 0.2, 0.0, 1.0)
        self.is_colliding = True

    def get_aabb(self) -> AABB:
        half = Vec3(self.width / 2.0, self.depth / 2.0, self.height / 2.0)
        return AABB(self.position - half, self.position + half)


class BVHNode:
    def __init__(self) -> None:
        self.left: Optional["BVHNode"] = None
        self.right: Optional["BVHNode"] = None
        self.box_id: int = -1
        self.aabb = AABB(
            Vec3(float("inf"), float("inf"), float("inf")),
            Vec3(float("-inf"), float("-inf"), float("-inf")),
        )

    def is_leaf(self) -> bool:
        return self.box_id != -1


def get_split_position(nodes: List[Tuple[int, int]], begin: int, end: int) -> int:
    return (begin + end) // 2


def create_leaf(box_id: int, box: Box) -> BVHNode:
    node = BVHNode()
    aabb = box.get_aabb()
    node.aabb.minimum = aabb.minimum
    node.aabb.maximum = aabb.maximum
    node.box_id = box_id
    return node


def create_subtree(
    nodes: List[Tuple[int, int]], begin: int, end: int, boxes: List[Box]
) -> BVHNode:
    if begin == end:
        return create_leaf(nodes[begin][0], boxes[nodes[begin][0]])

    split = get_split_position(nodes, begin, end)
    node = BVHNode()
    node.left = create_subtree(nodes, begin, split, boxes)
    node.right = create_subtree(nodes, split + 1, end, boxes)

    node.aabb.minimum = Vec3(
        min(node.left.aabb.minimum.x, node.right.aabb.minimum.x),
        min(node.left.aabb.minimum.y, node.right.aabb.minimum.y),
        min(node.left.aabb.minimum.z, node.right.aabb.minimum.z),
    )
    node.aabb.maximum = Vec3(
        max(node.left.aabb.maximum.x, node.right.aabb.maximum.x),
        max(node.left.aabb.maximum.y, node.right.aabb.maximum.y),
        max(node.left.aabb.maximum.z, node.right.aabb.maximum.z),
    )
    return node


def create_tree(boxes: List[Box]) -> Optional[BVHNode]:
    if not boxes:
        return None

    morton_list = [
        (index, calculate_morton_code(box.position.x, box.position.y, box.position.z))
        for index, box in enumerate(boxes)
    ]
    morton_list.sort(key=lambda item: item[1])
    return create_subtree(morton_list, 0, len(morton_list) - 1, boxes)


def check_collisions_bvh(
    root: Optional[BVHNode], aabbs: List[AABB]
) -> Tuple[List[Tuple[int, int]], int]:
    if root is None:
        return [], 0

    collisions: List[Tuple[int, int]] = []
    checks = 0
    stack: List[Tuple[BVHNode, BVHNode]] = [(root, root)]

    while stack:
        node_a, node_b = stack.pop()
        checks += 1

        if not aabb_intersect(node_a.aabb, node_b.aabb):
            continue

        if node_a.is_leaf() and node_b.is_leaf():
            idx_a = node_a.box_id
            idx_b = node_b.box_id
            if idx_a >= idx_b:
                continue
            if aabb_intersect(aabbs[idx_a], aabbs[idx_b]):
                collisions.append((idx_a, idx_b))
            continue

        if node_a.is_leaf():
            if node_b.left is not None:
                stack.append((node_a, node_b.left))
            if node_b.right is not None:
                stack.append((node_a, node_b.right))
        elif node_b.is_leaf():
            if node_a.left is not None:
                stack.append((node_a.left, node_b))
            if node_a.right is not None:
                stack.append((node_a.right, node_b))
        else:
            if node_a.left is not None and node_b.left is not None:
                stack.append((node_a.left, node_b.left))
            if node_a.left is not None and node_b.right is not None:
                stack.append((node_a.left, node_b.right))
            if node_a.right is not None and node_b.left is not None:
                stack.append((node_a.right, node_b.left))
            if node_a.right is not None and node_b.right is not None:
                stack.append((node_a.right, node_b.right))

    return collisions, checks


def compute_aabbs(boxes: List[Box]) -> List[AABB]:
    return [box.get_aabb() for box in boxes]


def check_collisions_bruteforce(aabbs: List[AABB]) -> Tuple[List[Tuple[int, int]], int]:
    collisions: List[Tuple[int, int]] = []
    checks = 0
    count = len(aabbs)
    for i in range(count):
        a = aabbs[i]
        for j in range(i + 1, count):
            b = aabbs[j]
            checks += 1
            if aabb_intersect(a, b):
                collisions.append((i, j))
    return collisions, checks


def check_collisions_sweep_and_prune(aabbs: List[AABB]) -> Tuple[List[Tuple[int, int]], int]:
    entries: List[Tuple[int, float, float, float, float, float, float]] = []
    for idx, aabb in enumerate(aabbs):
        entries.append(
            (
                idx,
                aabb.minimum.x,
                aabb.maximum.x,
                aabb.minimum.y,
                aabb.maximum.y,
                aabb.minimum.z,
                aabb.maximum.z,
            )
        )

    entries.sort(key=lambda entry: entry[1])

    collisions: List[Tuple[int, int]] = []
    checks = 0

    for i in range(len(entries)):
        idx_a, min_x_a, max_x_a, min_y_a, max_y_a, min_z_a, max_z_a = entries[i]
        for j in range(i + 1, len(entries)):
            idx_b, min_x_b, max_x_b, min_y_b, max_y_b, min_z_b, max_z_b = entries[j]

            if min_x_b > max_x_a:
                break

            checks += 1

            if (
                min_y_a <= max_y_b
                and max_y_a >= min_y_b
                and min_z_a <= max_z_b
                and max_z_a >= min_z_b
            ):
                collisions.append((idx_a, idx_b))

    return collisions, checks


class CollisionComparisonApp(ShowBase):
    def __init__(self) -> None:
        ShowBase.__init__(self)
        self.setBackgroundColor(0.07, 0.07, 0.07, 1.0)
        self.disableMouse()
        self.camera.setPos(70.0, -70.0, 40.0)
        self.camera.lookAt(0.0, 0.0, 0.0)
        self.camLens.setFov(45.0)
        self.camLens.setNearFar(0.1, 1000.0)

        self._unit_box = build_unit_box()

        self._setup_lighting()
        self._setup_environment()

        self.boxes = [
            Box(index, self.render, self._unit_box)
            for index in range(BOX_COUNT)
        ]

        info_pos = (-1.33, 0.72)
        self.info_text = OnscreenText(
            text="",
            parent=self.aspect2d,
            pos=info_pos,
            align=TextNode.A_left,
            scale=0.045,
            fg=(1, 1, 1, 1),
            mayChange=True,
        )

        self.algorithms = ("BVH", "Sweep and Prune", "Brute Force")
        self.current_algorithm_index = 0
        self.algorithm_button = DirectButton(
            text=self._algorithm_label(),
            scale=0.05,
            pos=(-1.33, 0.0, 0.87),
            parent=self.aspect2d,
            frameColor=(0.12, 0.12, 0.12, 0.85),
            text_align=TextNode.A_left,
            text_scale=0.9,
            text_fg=(1, 1, 1, 1),
            pad=(0.25, 0.15),
            command=self._cycle_algorithm,
        )

        self.camera_angle = 0.0
        self.camera_radius = 85.0
        self.camera_height = 40.0

        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0
        self.current_fps = 0

        self.taskMgr.add(self.update, "update")

    def _algorithm_label(self) -> str:
        return f"Algorithm: {self.algorithms[self.current_algorithm_index]}"

    def _cycle_algorithm(self) -> None:
        self.current_algorithm_index = (self.current_algorithm_index + 1) % len(self.algorithms)
        self.algorithm_button["text"] = self._algorithm_label()

    def _setup_lighting(self) -> None:
        self.render.clearLight()

        ambient = AmbientLight("ambient")
        ambient.set_color((0.2, 0.2, 0.2, 1.0))
        ambient_np = self.render.attach_new_node(ambient)
        self.render.setLight(ambient_np)

        main_directional = DirectionalLight("main_light")
        main_directional.set_color((0.8, 0.8, 0.8, 1.0))
        main_directional.setShadowCaster(True, 2048, 2048)
        main_np = self.render.attach_new_node(main_directional)
        main_np.setHpr(-45.0, -60.0, 0.0)
        self.render.setLight(main_np)

        fill_directional = DirectionalLight("fill_light")
        fill_directional.set_color((0.4, 0.4, 0.7, 1.0))
        fill_np = self.render.attach_new_node(fill_directional)
        fill_np.setHpr(135.0, -30.0, 0.0)
        self.render.setLight(fill_np)

        rim_directional = DirectionalLight("rim_light")
        rim_directional.set_color((0.7, 0.4, 0.4, 1.0))
        rim_np = self.render.attach_new_node(rim_directional)
        rim_np.setHpr(0.0, 60.0, 180.0)
        self.render.setLight(rim_np)

        self.render.setShaderAuto()

    def _setup_environment(self) -> None:
        fog = Fog("scene_fog")
        fog.set_color(0.07, 0.07, 0.07)
        fog.set_exp_density(0.0025)
        self.render.setFog(fog)

        card = CardMaker("floor")
        card.setFrame(-WORLD_SIZE, WORLD_SIZE, -WORLD_SIZE, WORLD_SIZE)
        floor = self.render.attach_new_node(card.generate())
        floor.setHpr(0.0, -90.0, 0.0)
        floor.setPos(0.0, 0.0, -WORLD_SIZE / 2.0)
        floor.setColor(0.15, 0.15, 0.15, 1.0)
        floor.setShaderAuto()

        grid = LineSegs()
        grid.setThickness(1.0)
        grid_color = (0.27, 0.27, 0.27, 1.0)
        grid.setColor(*grid_color)
        step = WORLD_SIZE / 10.0
        z_level = -WORLD_SIZE / 2.0 + 0.01
        for i in range(-10, 11):
            offset = i * step
            grid.moveTo(-WORLD_SIZE, offset, z_level)
            grid.drawTo(WORLD_SIZE, offset, z_level)
            grid.moveTo(offset, -WORLD_SIZE, z_level)
            grid.drawTo(offset, WORLD_SIZE, z_level)
        grid_node = grid.create()
        self.render.attach_new_node(grid_node)

        cube_lines = LineSegs()
        cube_lines.setThickness(2.0)
        cube_lines.setColor(0.27, 0.27, 0.27, 1.0)
        h = WORLD_SIZE / 2.0
        corners = [
            Vec3(-h, -h, -h),
            Vec3(h, -h, -h),
            Vec3(h, h, -h),
            Vec3(-h, h, -h),
            Vec3(-h, -h, h),
            Vec3(h, -h, h),
            Vec3(h, h, h),
            Vec3(-h, h, h),
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for start_idx, end_idx in edges:
            cube_lines.moveTo(corners[start_idx])
            cube_lines.drawTo(corners[end_idx])

        cube_node = cube_lines.create()
        cube_np = self.render.attach_new_node(cube_node)
        cube_np.setTransparency(True)
        cube_np.setColor(0.27, 0.27, 0.27, 0.2)

    def _update_camera(self, dt: float) -> None:
        self.camera_angle += dt * 0.06
        self.camera.setPos(
            math.sin(self.camera_angle) * self.camera_radius,
            -math.cos(self.camera_angle) * self.camera_radius,
            self.camera_height,
        )
        self.camera.lookAt(0.0, 0.0, 0.0)

    def _update_fps(self) -> None:
        self.frame_counter += 1
        now = time.perf_counter()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.current_fps = int(round(self.frame_counter / elapsed))
            self.frame_counter = 0
            self.last_fps_time = now

    def update(self, task) -> int:
        dt = globalClock.get_dt()
        dt = min(dt, 0.2)

        for box in self.boxes:
            box.update(dt)

        algorithm = self.algorithms[self.current_algorithm_index]
        aabbs = compute_aabbs(self.boxes)

        build_duration = 0.0
        check_duration = 0.0
        collisions: List[Tuple[int, int]] = []
        check_count = 0

        if algorithm == "BVH":
            build_start = time.perf_counter()
            tree = create_tree(self.boxes)
            build_duration = (time.perf_counter() - build_start) * 1000.0

            check_start = time.perf_counter()
            collisions, check_count = check_collisions_bvh(tree, aabbs)
            check_duration = (time.perf_counter() - check_start) * 1000.0
        else:
            detect_start = time.perf_counter()
            if algorithm == "Sweep and Prune":
                collisions, check_count = check_collisions_sweep_and_prune(aabbs)
            else:
                collisions, check_count = check_collisions_bruteforce(aabbs)
            check_duration = (time.perf_counter() - detect_start) * 1000.0

        for a, b in collisions:
            self.boxes[a].mark_collision()
            self.boxes[b].mark_collision()

        self._update_fps()
        self._update_camera(dt)

        info_lines = [
            "3D Collision Detection Comparison",
            f"Algorithm: {algorithm}",
            f"Boxes: {len(self.boxes)}",
            f"Collisions: {len(collisions)}",
            f"Build Time: {build_duration:.2f} ms",
            f"Detection Time: {check_duration:.2f} ms",
            f"Pairs Checked: {check_count}",
            f"FPS: {self.current_fps}",
        ]
        self.info_text.setText("\n".join(info_lines))

        return Task.cont


if __name__ == "__main__":
    app = CollisionComparisonApp()
    app.run()
