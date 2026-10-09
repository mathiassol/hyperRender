"""Writes a block-compressed, mipmapped .dds next to every texture a glTF uses, which
Donut loads in place of the .jpg / .png (GltfImporter looks for one with the same name).

    python tools/compress_textures.py assets/downloaded/Sponza/glTF/Sponza.gltf

Base colour and emissive: BC7 sRGB, mips filtered in linear light. Normal maps: BC5 (x, y;
the shaders rebuild z). Metal-roughness, occlusion and the rest: BC7. A quarter of RGBA8's
memory, and nothing to decode or mip at load. A .dds newer than its source is kept.

Needs `pip install etcpak numpy pillow`. The scene builders (tools/make_*.py) run it after
their export.
"""
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

BC5_UNORM, BC7_UNORM, BC7_UNORM_SRGB = 83, 98, 99


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def mips(rgba, srgb):
    """The mip chain down to 1 x 1, each level max(1, size >> i) as GPUs count them, box
    filtered (in linear light for sRGB colour)."""
    level = rgba.astype(np.float32) / 255.0
    if srgb:
        level[..., :3] = srgb_to_linear(level[..., :3])
    out = []
    while True:
        img = level.copy()
        if srgb:
            img[..., :3] = linear_to_srgb(img[..., :3])
        out.append(np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
        h, w = level.shape[:2]
        if w == 1 and h == 1:
            return out
        size = (max(w // 2, 1), max(h // 2, 1))
        level = np.stack([np.asarray(Image.fromarray(level[..., c], "F").resize(size, Image.BOX)) for c in range(4)], -1)


def blocks(img):
    """Pads to whole 4 x 4 blocks by repeating the edge."""
    h, w = img.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return np.ascontiguousarray(img), img.shape[1], img.shape[0]


def compress(job):
    import etcpak
    src, dst, kind = job
    rgba = np.asarray(Image.open(src).convert("RGBA"))
    fmt = {"normal": BC5_UNORM, "srgb": BC7_UNORM_SRGB, "linear": BC7_UNORM}[kind]
    levels = mips(rgba, kind == "srgb")
    data = []
    for level in levels:
        img, w, h = blocks(level)
        data.append(etcpak.compress_bc5(img.tobytes(), w, h) if fmt == BC5_UNORM else etcpak.compress_bc7(img.tobytes(), w, h, None))
    h, w = rgba.shape[:2]
    # DDS_HEADER (caps, height, width, mips, a FourCC "DX10" pixel format) + DDS_HEADER_DXT10.
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
    pixel_format = struct.pack("<II4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    header = struct.pack("<7I44x", 124, flags, h, w, len(data[0]), 0, len(levels)) + pixel_format \
        + struct.pack("<5I", 0x1000 | 0x400000 | 0x8, 0, 0, 0, 0)
    dx10 = struct.pack("<5I", fmt, 3, 0, 1, 0)
    with open(dst + ".tmp", "wb") as f:
        f.write(b"DDS " + header + dx10 + b"".join(data))
    os.replace(dst + ".tmp", dst)
    return dst


def jobs(gltf_path):
    """(source, .dds, kind) for every image the materials use, by what they use it for."""
    gltf = json.load(open(gltf_path))
    folder = os.path.dirname(gltf_path)
    kinds = {}

    def use(info, kind):
        if info is None:
            return
        image = gltf["textures"][info["index"]].get("source")
        if image is not None:
            kinds.setdefault(image, kind)

    for m in gltf.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        use(m.get("normalTexture"), "normal")
        use(pbr.get("baseColorTexture"), "srgb")
        use(m.get("emissiveTexture"), "srgb")
        use(pbr.get("metallicRoughnessTexture"), "linear")
        use(m.get("occlusionTexture"), "linear")
        for ext in m.get("extensions", {}).values():
            for key, value in ext.items():
                if key.endswith("Texture") and isinstance(value, dict):
                    use(value, "linear")
    # Donut looks for name.dds for name.jpg and name.png alike: images sharing a name get
    # none (one .dds would stand in for both).
    stems = [os.path.splitext(im.get("uri", ""))[0] for im in gltf.get("images", [])]
    out = []
    for index, kind in kinds.items():
        uri = gltf["images"][index].get("uri")
        if not uri:
            continue   # embedded: Donut only looks beside files
        src = os.path.join(folder, uri.replace("%20", " "))
        dst = os.path.splitext(src)[0] + ".dds"
        if stems.count(os.path.splitext(uri)[0]) > 1:
            print(f"compress_textures: {uri} shares its name with another image; left as it is")
            if os.path.exists(dst):
                os.remove(dst)
            continue
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        out.append((src, dst, kind))
    return out


def main(paths):
    work = [j for p in paths for j in jobs(p)]
    if not work:
        print("compress_textures: all up to date")
        return
    with ProcessPoolExecutor() as pool:
        for i, dst in enumerate(pool.map(compress, work)):
            print(f"compress_textures: [{i + 1}/{len(work)}] {os.path.basename(dst)}")


if __name__ == "__main__":
    main(sys.argv[1:])
