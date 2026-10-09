"""Mean colour of PNG captures, whole image or a region, without dependencies.

    python tools/png_stats.py a.png [b.png ...] [--rect x0 y0 x1 y1]
    python tools/png_stats.py --diff a.png b.png      mean / max absolute difference
    python tools/png_stats.py --thumb in.png out.png  4x box-downscaled copy (a reference)
    python tools/png_stats.py --crop in.png out.png x0 y0 x1 y1   a region, at full resolution
    python tools/png_stats.py --check ref.png in.png  compare in.png, downscaled, with ref.png;
                                                      exit 1 past mean 0.5 or any pixel 24 (of 255)

For comparing captures (e.g. denoised vs raw, module on vs off). Supports 8-bit
RGB / RGBA, non-interlaced, which is what hyper's --capture writes.
"""
import struct
import sys
import zlib


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, width, height, channels = 8, b"", 0, 0, 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and colour in (2, 6), "8-bit RGB/RGBA only"
            channels = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, prev = [], bytearray(stride)
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return width, height, channels, rows


def diff(a, b):
    wa, ha, ca, ra = read_png(a)
    wb, hb, cb, rb = read_png(b)
    assert (wa, ha) == (wb, hb), "sizes differ"
    total, worst, changed = 0, 0, 0
    for y in range(ha):
        for x in range(wa):
            d = max(abs(ra[y][x * ca + k] - rb[y][x * cb + k]) for k in range(3))
            total += d
            worst = max(worst, d)
            changed += d > 0
    print(f"mean abs diff {total / (wa * ha):.4f}, max {worst}, pixels changed {changed} / {wa * ha}")


def thumbnail(path, factor):
    """Box-filtered RGB rows, factor x smaller (the remainder is dropped)."""
    w, h, ch, rows = read_png(path)
    tw, th = w // factor, h // factor
    n = factor * factor
    out = []
    for ty in range(th):
        sums = [0] * (tw * 3)
        for row in rows[ty * factor:(ty + 1) * factor]:
            for tx in range(tw):
                for x in range(tx * factor, (tx + 1) * factor):
                    for k in range(3):
                        sums[tx * 3 + k] += row[x * ch + k]
        out.append(bytearray((s + n // 2) // n for s in sums))
    return tw, th, out


def write_png(path, w, h, rows):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def check(reference, capture, factor=4, max_mean=0.5, max_pixel=24):
    """Compares a capture, downscaled, with a reference thumbnail. True if it matches."""
    tw, th, thumb = thumbnail(capture, factor)
    rw, rh, rc, ref = read_png(reference)
    if (rw, rh) != (tw, th):
        print(f"size {tw}x{th} vs reference {rw}x{rh}")
        return False
    total, worst = 0, 0
    for y in range(th):
        for x in range(tw):
            d = max(abs(thumb[y][x * 3 + k] - ref[y][x * rc + k]) for k in range(3))
            total += d
            worst = max(worst, d)
    mean = total / (tw * th)
    print(f"mean {mean:.3f}, max {worst}")
    return mean <= max_mean and worst <= max_pixel


def main():
    args = sys.argv[1:]
    if args and args[0] == "--diff":
        diff(args[1], args[2])
        return
    if args and args[0] == "--thumb":
        w, h, rows = thumbnail(args[1], 4)
        write_png(args[2], w, h, rows)
        return
    if args and args[0] == "--crop":
        w, h, ch, rows = read_png(args[1])
        x0, y0, x1, y1 = (int(v) for v in args[3:7])
        x1, y1 = min(x1, w), min(y1, h)
        out = [bytearray(b for x in range(x0, x1) for b in rows[y][x * ch:x * ch + 3]) for y in range(y0, y1)]
        write_png(args[2], x1 - x0, y1 - y0, out)
        return
    if args and args[0] == "--check":
        sys.exit(0 if check(args[1], args[2]) else 1)
    rect = None
    if "--rect" in args:
        i = args.index("--rect")
        rect = [int(v) for v in args[i + 1:i + 5]]
        args = args[:i] + args[i + 5:]
    for path in args:
        w, h, ch, rows = read_png(path)
        x0, y0, x1, y1 = rect or (0, 0, w, h)
        total, n = [0, 0, 0], 0
        for y in range(y0, min(y1, h)):
            row = rows[y]
            for x in range(x0, min(x1, w)):
                for k in range(3):
                    total[k] += row[x * ch + k]
                n += 1
        mean = [t / max(n, 1) for t in total]
        print(f"{path}: mean rgb {mean[0]:.1f} {mean[1]:.1f} {mean[2]:.1f}")


if __name__ == "__main__":
    main()
