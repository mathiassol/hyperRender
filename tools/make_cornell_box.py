"""Writes assets/scenes/cornell_box.gltf: a closed Cornell box (5.5 m, deepened
to 11 m so the camera can frame it) lit only by an emissive ceiling panel, with
a camera inside. No sun or sky reaches in,
so everything but the panel itself is lit by global illumination.

    python tools/make_cornell_box.py
"""
import base64
import json
import math
import os
import struct

WHITE = (0.73, 0.73, 0.73)
RED = (0.63, 0.065, 0.05)
GREEN = (0.14, 0.45, 0.09)
S = 5.5  # room width and height, metres
D = 11.0  # depth: the camera stands back from the classic box, inside the closed room


def quad(corners, normal):
    """Two triangles over 4 corners, wound counter-clockwise around normal."""
    p0, p1, p2 = corners[0], corners[1], corners[2]
    e1 = [p1[i] - p0[i] for i in range(3)]
    e2 = [p2[i] - p0[i] for i in range(3)]
    c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    if sum(c[i] * normal[i] for i in range(3)) < 0:
        corners = corners[::-1]
    return [(p, normal) for p in corners]


def box(centre, size, angle_deg):
    """The five visible faces of a block (no bottom), rotated about y."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hx, hy, hz = size[0] / 2, size[1], size[2] / 2

    def p(x, y, z):
        return (centre[0] + x * ca + z * sa, y, centre[2] - x * sa + z * ca)

    def n(x, z):
        return (x * ca + z * sa, 0.0, -x * sa + z * ca)

    faces = [
        quad([p(-hx, hy, -hz), p(hx, hy, -hz), p(hx, hy, hz), p(-hx, hy, hz)], (0.0, 1.0, 0.0)),
        quad([p(-hx, 0, hz), p(hx, 0, hz), p(hx, hy, hz), p(-hx, hy, hz)], n(0, 1)),
        quad([p(-hx, 0, -hz), p(hx, 0, -hz), p(hx, hy, -hz), p(-hx, hy, -hz)], n(0, -1)),
        quad([p(hx, 0, -hz), p(hx, 0, hz), p(hx, hy, hz), p(hx, hy, -hz)], n(1, 0)),
        quad([p(-hx, 0, -hz), p(-hx, 0, hz), p(-hx, hy, hz), p(-hx, hy, -hz)], n(-1, 0)),
    ]
    return faces


def main():
    # Room faces point inwards.
    white = [
        quad([(0, 0, 0), (S, 0, 0), (S, 0, D), (0, 0, D)], (0.0, 1.0, 0.0)),      # floor
        quad([(0, S, 0), (S, S, 0), (S, S, D), (0, S, D)], (0.0, -1.0, 0.0)),     # ceiling
        quad([(0, 0, 0), (S, 0, 0), (S, S, 0), (0, S, 0)], (0.0, 0.0, 1.0)),      # back
        quad([(0, 0, D), (S, 0, D), (S, S, D), (0, S, D)], (0.0, 0.0, -1.0)),     # front (behind the camera)
    ] + box((1.85, 0, 2.0), (1.65, 3.3, 1.65), 17) + box((3.7, 0, 3.6), (1.65, 1.65, 1.65), -17)
    red = [quad([(0, 0, 0), (0, 0, D), (0, S, D), (0, S, 0)], (1.0, 0.0, 0.0))]
    green = [quad([(S, 0, 0), (S, 0, D), (S, S, D), (S, S, 0)], (-1.0, 0.0, 0.0))]
    y = S - 0.01
    light = [quad([(2.1, y, 2.2), (3.4, y, 2.2), (3.4, y, 3.3), (2.1, y, 3.3)], (0.0, -1.0, 0.0))]

    groups = [("white", white), ("red", red), ("green", green), ("light", light)]
    blob = bytearray()
    accessors, views, primitives = [], [], []
    for material, faces in enumerate(g[1] for g in groups):
        verts = [v for f in faces for v in f]
        indices = []
        for i in range(len(faces)):
            b = i * 4
            indices += [b, b + 1, b + 2, b, b + 2, b + 3]
        positions = b"".join(struct.pack("<3f", *v[0]) for v in verts)
        normals = b"".join(struct.pack("<3f", *v[1]) for v in verts)
        index_bytes = b"".join(struct.pack("<I", i) for i in indices)
        lo = [min(v[0][k] for v in verts) for k in range(3)]
        hi = [max(v[0][k] for v in verts) for k in range(3)]

        first = len(accessors)
        for data, target in ((positions, 34962), (normals, 34962), (index_bytes, 34963)):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
            blob += data
            while len(blob) % 4:
                blob += b"\0"
        accessors.append({"bufferView": first, "componentType": 5126, "count": len(verts), "type": "VEC3", "min": lo, "max": hi})
        accessors.append({"bufferView": first + 1, "componentType": 5126, "count": len(verts), "type": "VEC3"})
        accessors.append({"bufferView": first + 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"})
        primitives.append({"attributes": {"POSITION": first, "NORMAL": first + 1}, "indices": first + 2, "material": material})

    def mat(name, colour, emissive=None):
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*colour, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if emissive:
            m["emissiveFactor"] = list(emissive[0])
            m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": emissive[1]}}
        return m

    gltf = {
        "asset": {"version": "2.0", "generator": "hyperRender tools/make_cornell_box.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "CornellBox", "mesh": 0},
            {"name": "Camera", "camera": 0, "translation": [S / 2, S / 2, D - 0.2]},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.62, "znear": 0.05, "aspectRatio": 1.37}}],
        "meshes": [{"name": "CornellBox", "primitives": primitives}],
        "materials": [
            mat("white", WHITE), mat("red", RED), mat("green", GREEN),
            mat("light", (0.78, 0.78, 0.78), ((1.0, 0.72, 0.35), 17.0)),
        ],
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
        "bufferViews": views,
        "accessors": accessors,
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "assets", "scenes", "cornell_box.gltf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
