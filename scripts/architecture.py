"""Draw the CNN-LSTM pipeline figure used on the project page and in the code README.

    python scripts/architecture.py

writes static/images/architecture.svg (one row, desktop) and
static/images/architecture-vertical.svg (stacked, narrow screens).
Blocks are placed from measured label widths so text never overlaps.
"""
from pathlib import Path

import numpy as np
from PIL import ImageFont

OUT = Path(__file__).resolve().parent.parent / "static" / "images"
FONTS = "/usr/share/fonts/truetype/liberation/LiberationSans-{}.ttf"

INK, MUTED, ARROW = "#1f2328", "#59636e", "#8c959f"
N_FILL, N_STROKE = "#f6f8fa", "#afb8c1"
I_COL, Q_COL = "#2f6fdb", "#e07b39"
CNN_FILL, CNN_STROKE, CNN_TEXT = "#eaf1fb", "#6b93d6", "#2f5fa8"
LSTM_FILL, LSTM_STROKE, LSTM_CELL = "#fdf3e4", "#d39b4a", "#f7deb8"
FC_FILL, FC_STROKE, FC_LINK = "#e9f5ee", "#5aa57a", "#f1f8f3"
OUT_BAR, OUT_HI = "#d0d7de", "#24292f"


def f(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def rect(x, y, w, h, fill, stroke=None, rx=0, sw=1.25):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    r = f' rx="{f(rx)}"' if rx else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{r} fill="{fill}"{s}/>'


# Each block draws itself with its left edge at x0 and its centre line at cy.

def iq_frame(x0, cy):
    w, h = 116, 80
    out = [rect(x0, cy - h / 2, w, h, N_FILL, N_STROKE, rx=6)]
    rng = np.random.default_rng(5)
    nsym, sps = 14, 8
    levels = np.array([-3, -1, 1, 3]) / 3
    sym = rng.choice(levels, nsym) + 1j * rng.choice(levels, nsym)
    up = np.zeros(nsym * sps, complex)
    up[::sps] = sym
    t = np.arange(-3 * sps, 3 * sps + 1) / sps
    x = np.convolve(up, np.sinc(t) * np.hanning(t.size), mode="same")
    i_part, q_part = np.real(x), np.imag(x)
    scale = max(np.abs(i_part).max(), np.abs(q_part).max())
    xs = np.linspace(x0 + 10, x0 + w - 10, x.size)
    for comp, col in ((i_part / scale, I_COL), (q_part / scale, Q_COL)):
        pts = " ".join(f"{f(a)},{f(cy - 27 * b)}" for a, b in zip(xs, comp))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
    return out


def windows(x0, cy):
    w = 120
    seg, step = w * 224 / 1024, w * 112 / 1024  # to scale: 224-sample windows, 50% overlap
    out = [rect(x0, cy - 42, w, 6, "#d0d7de", rx=3)]
    out += [rect(x0 + i * step, cy - 28 + i * 9, seg, 6, ARROW, rx=3) for i in range(8)]
    return out


def input_image(x0, cy):
    s = 56
    out = [rect(x0 + 16, cy - 36, s, s, "#eef1f4", N_STROKE, rx=3),
           rect(x0 + 8, cy - 28, s, s, "#e3e8ed", N_STROKE, rx=3)]
    bands = 0.5 + 0.5 * np.sin(np.linspace(0, 5.2, 14)) * np.cos(np.linspace(0, 2.1, 14))
    for k, b in enumerate(bands):
        g = int(0xc4 + (0xf1 - 0xc4) * b)
        out.append(rect(x0, cy - 20 + k * s / 14, s, s / 14 + 0.3, f"#{g:02x}{g + 4:02x}{g + 8:02x}"))
    out.append(rect(x0, cy - 20, s, s, "none", N_STROKE, rx=3))
    out.append(f'<text x="{f(x0 + 76)}" y="{f(cy - 30)}" class="d" style="text-anchor: start">×8</text>')
    return out


def cnn(x0, cy):
    w = 108
    return [f'<polygon points="{f(x0)},{f(cy - 48)} {f(x0 + w)},{f(cy - 20)} {f(x0 + w)},{f(cy + 20)} '
            f'{f(x0)},{f(cy + 48)}" fill="{CNN_FILL}" stroke="{CNN_STROKE}" stroke-width="1.25" '
            'stroke-linejoin="round"/>',
            f'<text x="{f(x0 + 48)}" y="{f(cy + 5)}" class="t" fill="{CNN_TEXT}">CNN</text>']


def lstm(x0, cy):
    w, cw, n = 136, 12, 8
    gap = (w - 20 - n * cw) / (n - 1)
    out = [rect(x0, cy - 28, w, 56, LSTM_FILL, LSTM_STROKE, rx=8)]
    for i in range(n):
        cx = x0 + 10 + i * (cw + gap)
        out.append(rect(cx, cy - 13, cw, 26, LSTM_CELL, LSTM_STROKE, rx=3, sw=1))
        if i < n - 1:
            out.append(f'<line x1="{f(cx + cw)}" y1="{f(cy)}" x2="{f(cx + cw + gap)}" y2="{f(cy)}" '
                       f'stroke="{LSTM_STROKE}" stroke-width="1.2"/>')
    return out


def dense(x0, cy):
    bars = [(x0, cy - 50, 100), (x0 + 30, cy - 31, 62), (x0 + 60, cy - 21, 42)]  # 2048 -> 512 -> 256
    out = [f'<polygon points="{f(xa + 13)},{f(ya)} {f(xb)},{f(yb)} {f(xb)},{f(yb + hb)} '
           f'{f(xa + 13)},{f(ya + ha)}" fill="{FC_LINK}"/>'
           for (xa, ya, ha), (xb, yb, hb) in zip(bars, bars[1:])]
    out += [rect(x, y, 13, h, FC_FILL, FC_STROKE, rx=3) for x, y, h in bars]
    return out


def prediction(x0, cy):
    lengths = [9, 6, 12, 15, 8, 17, 11, 62, 7]
    return [rect(x0, cy - 43 + k * 10, ln, 6, OUT_HI if ln == max(lengths) else OUT_BAR, rx=2)
            for k, ln in enumerate(lengths)]


# draw function, width, half-height, title, detail lines
BLOCKS = [
    (iq_frame, 116, 40, "I/Q frame", ["1024 samples"]),
    (windows, 120, 42, "Windowing", ["8 × 224 samples", "50% overlap"]),
    (input_image, 72, 36, "Input image", ["3 × 224 × 224", "I/Q, amplitude, phase"]),
    (cnn, 108, 48, "Modified AlexNet", ["5 conv + GAP → 256", "shared weights"]),
    (lstm, 136, 28, "LSTM", ["8 steps", "hidden size 256"]),
    (dense, 73, 50, "Fully connected", ["2048 → 512 → 256"]),
    (prediction, 62, 43, "Prediction", ["9 classes"]),
]


def label_width(title, lines, t_size, d_size):
    bold = ImageFont.truetype(FONTS.format("Bold"), t_size)
    reg = ImageFont.truetype(FONTS.format("Regular"), d_size)
    return 1.06 * max([bold.getlength(title)] + [reg.getlength(s) for s in lines])


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{ARROW}" '
            'stroke-width="1.5" marker-end="url(#arrow)"/>')


def text(x, y, cls, s, anchor=None):
    a = f' style="text-anchor: {anchor}"' if anchor else ""
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}"{a}>{s}</text>'


def wide(t_size=15, d_size=13, margin=20, min_gap=28, label_gap=14):
    cy, el = 76, []
    x0 = w = c = lw = 0.0
    prev_draw = None
    for draw, bw, hh, title, lines in BLOCKS:
        blw = label_width(title, lines, t_size, d_size)
        if prev_draw is None:
            bx0 = margin + max(0, (blw - bw) / 2)
        else:
            bx0 = max(x0 + w + min_gap, c + lw / 2 + label_gap + blw / 2 - bw / 2)
            start = x0 + w + 6 + (24 if prev_draw is input_image else 0)  # clear the "×8" tag
            el.append(arrow(start, cy, bx0 - 8, cy))
        x0, w, lw, prev_draw = bx0, bw, blw, draw
        c = x0 + w / 2
        el += draw(x0, cy)
        el.append(text(c, cy + 80, "t", title))
        el += [text(c, cy + 100 + 17 * k, "d", s) for k, s in enumerate(lines)]
    width = max(x0 + w, c + lw / 2) + margin
    return (0, 0, width, cy + 100 + 17 + 18), el, t_size, d_size


def vertical(t_size=16, d_size=14, margin=20, gap=34):
    col = margin + max(w for _, w, *_ in BLOCKS) / 2
    label_x = margin + max(w for _, w, *_ in BLOCKS) + 28
    el, y, rows = [], margin, []
    for draw, w, hh, title, lines in BLOCKS:
        cy = y + hh
        el += draw(col - w / 2, cy)
        top = cy - (20 + 19 * len(lines)) / 2 + 15
        el.append(text(label_x, top, "t", title, "start"))
        el += [text(label_x, top + 21 + 19 * k, "d", s, "start") for k, s in enumerate(lines)]
        rows.append((y, y + 2 * hh))
        y += 2 * hh + gap
    el += [arrow(col, b + 6, col, t - 8) for (_, b), (t, _) in zip(rows, rows[1:])]
    width = label_x + max(label_width(t, ls, t_size, d_size) for *_, t, ls in BLOCKS) + margin
    return (0, 0, width, y - gap + margin), el, t_size, d_size


def write(name, layout):
    (vx, vy, vw, vh), el, t_size, d_size = layout()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{f(vw)}" height="{f(vh)}" viewBox="{f(vx)} {f(vy)} {f(vw)} {f(vh)}" role="img" aria-labelledby="title">
<title id="title">CNN-LSTM architecture: I/Q frame, eight overlapping windows, 3-channel input image, modified AlexNet, LSTM, fully connected layers, prediction over nine classes</title>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/>
</marker>
<style>
text {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; text-anchor: middle; }}
.t {{ font-size: {t_size}px; font-weight: 600; fill: {INK}; }}
.d {{ font-size: {d_size}px; fill: {MUTED}; }}
</style>
</defs>
<rect x="{f(vx)}" y="{f(vy)}" width="{f(vw)}" height="{f(vh)}" fill="#fff"/>
{chr(10).join(el)}
</svg>
"""
    (OUT / name).write_text(svg)
    print(f"{name}: {f(vw)} x {f(vh)}")


if __name__ == "__main__":
    write("architecture.svg", wide)
    write("architecture-vertical.svg", vertical)
