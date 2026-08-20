r"""Generate the desktop icon.

The app's visual identity is the neumorphic orb, so that is what the icon is: a
soft indigo tile with a lit sphere and two sound arcs coming off it. The arcs
say "this listens and talks" in a way a plain circle does not.

An icon has to survive being drawn at 16 pixels, which is where most of the
design decisions come from — no thin strokes, no gradients that flatten to mud,
and a silhouette that is still readable when every interior detail is gone. The
arcs are dropped below 32 px for exactly that reason; at that size they turn
into grey fringing around the sphere and make it look blurry rather than round.

Rendered at 8x and downsampled with Lanczos, because PIL has no antialiased
circle. Writes every size Windows asks for into one .ico.

Run:  python scripts\make_icon.py          (needs Pillow)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "coach.ico"

# The palette is the app's own, so the icon and the window agree.
INDIGO = (88, 102, 224)
INDIGO_DEEP = (62, 74, 186)
SURFACE = (233, 237, 245)
WHITE = (255, 255, 255)

SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
SS = 8                      # supersampling factor


def tile(n: int) -> Image.Image:
    """The rounded background, as a smooth vertical gradient.

    Stacking two rounded rectangles was tried first and left a hard horizontal
    seam plus a ghost corner where the upper one ended — at 256 px it read as a
    rendering bug rather than as depth. A per-row interpolation has no edge to
    give away.
    """
    grad = Image.new("RGB", (1, n))
    for y in range(n):
        t = y / max(1, n - 1)
        grad.putpixel((0, y), tuple(
            round(a + (b - a) * t) for a, b in zip(INDIGO, INDIGO_DEEP)))
    grad = grad.resize((n, n))

    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, n - 1, n - 1), radius=int(n * 0.22), fill=255)

    out = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    out.paste(grad, (0, 0), mask)
    return out


def render(px: int) -> Image.Image:
    """One frame, drawn large and shrunk down."""
    n = px * SS
    img = tile(n)
    d = ImageDraw.Draw(img)

    detailed = px >= 32
    # With the arcs present the sphere sits left of centre to make room; alone,
    # it centres, or it looks like a mistake.
    cx = n * (0.42 if detailed else 0.5)
    cy = n * 0.5
    r = n * (0.22 if detailed else 0.27)

    # The sphere, lit from the upper left — the same lamp the CSS shadows in the
    # app assume, so the icon and the page look like the same object.
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=SURFACE)
    hr = r * 0.62
    hx, hy = cx - r * 0.30, cy - r * 0.32
    d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=WHITE)

    if detailed:
        # Two arcs: sound leaving the orb. Thick enough to survive the shrink.
        w = max(2, int(n * 0.045))
        for i, spread in enumerate((0.34, 0.50)):
            rad = n * spread
            d.arc((cx - rad, cy - rad, cx + rad, cy + rad),
                  start=-42, end=42, fill=WHITE, width=w)

    return img.resize((px, px), Image.LANCZOS)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(s) for s in SIZES]
    frames[-1].save(OUT, format="ICO",
                    sizes=[(s, s) for s in SIZES], append_images=frames[:-1])

    # A PNG preview too, so the icon can be checked without a file explorer.
    preview = Image.new("RGBA", (sum(s + 8 for s in SIZES), 264), (250, 250, 252, 255))
    x = 0
    for s, f in zip(SIZES, frames):
        preview.paste(f, (x, (264 - s) // 2), f)
        x += s + 8
    preview.save(OUT.with_name("coach_preview.png"))

    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
    print(f"sizes: {', '.join(str(s) for s in SIZES)}")
    print(f"preview: {OUT.with_name('coach_preview.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
