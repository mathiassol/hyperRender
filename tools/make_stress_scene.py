"""Stress scenes for the GPU-driven renderer, into assets/downloaded/.

    python tools/make_stress_scene.py [N]          stress_city.gltf: culling
    python tools/make_stress_scene.py field [N]    stress_field.gltf: cluster LOD

city: N x N buildings (boxes of random height) with a detailed sphere at every
crossing, plus a street-level camera. With the default N = 100 that is 20,000
instances, most of them hidden behind buildings from the street.

field: N x N dense spheres (36,864 triangles each) on open ground, seen from
above, so little is occluded and most spheres are small on screen. With the
default N = 60, 3,600 instances and 133M triangles.
"""
import base64
import json
import math
import os
import random
import struct
import sys


def box():
    """Unit box, y from 0 to 1, x/z from -0.5 to 0.5, outward normals."""
    verts, idx = [], []
    faces = [
        ((0, 1, 0), [(-.5, 1, -.5), (-.5, 1, .5), (.5, 1, .5), (.5, 1, -.5)]),
        ((0, -1, 0), [(-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5)]),
        ((1, 0, 0), [(.5, 0, -.5), (.5, 1, -.5), (.5, 1, .5), (.5, 0, .5)]),
        ((-1, 0, 0), [(-.5, 0, -.5), (-.5, 0, .5), (-.5, 1, .5), (-.5, 1, -.5)]),
        ((0, 0, 1), [(-.5, 0, .5), (.5, 0, .5), (.5, 1, .5), (-.5, 1, .5)]),
        ((0, 0, -1), [(-.5, 0, -.5), (-.5, 1, -.5), (.5, 1, -.5), (.5, 0, -.5)]),
    ]
    for n, corners in faces:
        b = len(verts)
        verts += [(c, n) for c in corners]
        idx += [b, b + 1, b + 2, b, b + 2, b + 3]
    return verts, idx


def sphere(stacks=48, slices=96):
    verts, idx = [], []
    for i in range(stacks + 1):
        v = i / stacks
        phi = v * math.pi
        for j in range(slices + 1):
            u = j / slices
            th = u * 2 * math.pi
            n = (math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th))
            verts.append((n, n))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx += [a, a + 1, b, a + 1, b + 1, b]
    return verts, idx


def main():
    args = sys.argv[1:]
    field = bool(args) and args[0] == "field"
    if field:
        args = args[1:]
    n = int(args[0]) if args else (60 if field else 100)
    spacing = 6.0 if field else 12.0
    random.seed(7)
    blob = bytearray()
    views, accessors, meshes = [], [], []

    def add_mesh(name, verts, idx, material):
        first = len(accessors)
        pos = b"".join(struct.pack("<3f", *v[0]) for v in verts)
        nrm = b"".join(struct.pack("<3f", *v[1]) for v in verts)
        ind = b"".join(struct.pack("<I", i) for i in idx)
        lo = [min(v[0][k] for v in verts) for k in range(3)]
        hi = [max(v[0][k] for v in verts) for k in range(3)]
        for data, target in ((pos, 34962), (nrm, 34962), (ind, 34963)):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
            blob.extend(data)
            while len(blob) % 4:
                blob.append(0)
        accessors.append({"bufferView": first, "componentType": 5126, "count": len(verts), "type": "VEC3", "min": lo, "max": hi})
        accessors.append({"bufferView": first + 1, "componentType": 5126, "count": len(verts), "type": "VEC3"})
        accessors.append({"bufferView": first + 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": first, "NORMAL": first + 1},
                                                     "indices": first + 2, "material": material}]})

    add_mesh("building", *box(), 0)
    add_mesh("ball", *(sphere(96, 192) if field else sphere()), 3 if field else 1)
    add_mesh("ground", [((-1, 0, -1), (0, 1, 0)), ((1, 0, -1), (0, 1, 0)), ((1, 0, 1), (0, 1, 0)), ((-1, 0, 1), (0, 1, 0))],
             [0, 2, 1, 0, 3, 2], 2)

    nodes = []
    half = n * spacing / 2
    for gx in range(n):
        for gz in range(n):
            if field:
                x = gx * spacing - half + spacing / 2
                z = gz * spacing - half + spacing / 2
                nodes.append({"mesh": 1, "translation": [x, 2.0, z], "scale": [2.0, 2.0, 2.0]})
                continue
            x = gx * spacing - half + spacing / 2
            z = gz * spacing - half + spacing / 2
            height = random.uniform(6, 40)
            nodes.append({"mesh": 0, "translation": [x, 0, z], "scale": [spacing * 0.6, height, spacing * 0.6]})
            nodes.append({"mesh": 1, "translation": [x + spacing / 2, 1.2, z + spacing / 2], "scale": [1.0, 1.0, 1.0]})
    nodes.append({"mesh": 2, "scale": [half + spacing, 1, half + spacing]})
    # Street level, looking down a street (between two building columns).
    # Streets run along x = k * spacing - half; the camera looks down one (-z) from the edge.
    if field:
        # Above one corner, looking across the field towards the far corner, pitched down.
        yaw, pitch = math.radians(-45), math.radians(-18)
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        q = [cy * sp, sy * cp, -sy * sp, cy * cp]   # yaw about y, then pitch about x
        nodes.append({"name": "Camera", "camera": 0, "translation": [-half + 4, 18.0, half - 4], "rotation": q})
    else:
        nodes.append({"name": "Camera", "camera": 0, "translation": [spacing * 20 - half, 2.0, half - 2.0]})
    scene_nodes = list(range(len(nodes)))

    gltf = {
        "asset": {"version": "2.0", "generator": "hyperRender tools/make_stress_scene.py"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.1, "aspectRatio": 1.37}}],
        "meshes": meshes,
        "materials": [
            {"name": "concrete", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.58, 0.55, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}},
            {"name": "chrome", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.92, 1], "metallicFactor": 1.0, "roughnessFactor": 0.2}},
            {"name": "asphalt", "pbrMetallicRoughness": {"baseColorFactor": [0.12, 0.12, 0.13, 1], "metallicFactor": 0.0, "roughnessFactor": 0.95}},
            {"name": "clay", "pbrMetallicRoughness": {"baseColorFactor": [0.75, 0.45, 0.3, 1], "metallicFactor": 0.0, "roughnessFactor": 0.5}},
        ],
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
        "bufferViews": views,
        "accessors": accessors,
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "assets", "downloaded", "stress_field.gltf" if field else "stress_city.gltf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gltf, f, separators=(",", ":"))
    tris = len((sphere(96, 192) if field else sphere())[1]) // 3
    total = n * n * tris if field else n * n * (tris + 12)
    print(f"wrote {out}: {len(nodes) - 1} instances, {total:,} triangles")


if __name__ == "__main__":
    main()
