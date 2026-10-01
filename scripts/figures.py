"""Draw the figures for the project page and the code README.

    python scripts/figures.py

Writes to static/images/:
    architecture.svg           pipeline, one row (desktop, README)
    architecture-vertical.svg  pipeline, stacked (narrow screens)
    equation.svg               amplitude and phase definitions (Method section)

All text is typeset by LaTeX (latex + dvisvgm, Latin Modern) and embedded as
glyph outlines, so the SVGs need no fonts. Blocks are placed from the measured
label widths, so labels never overlap.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "static" / "images"

INK, MUTED, ARROW = "#1f2328", "#59636e", "#8c959f"
N_FILL, N_STROKE = "#f6f8fa", "#afb8c1"
I_COL, Q_COL = "#2f6fdb", "#e07b39"
CNN_FILL, CNN_STROKE, CNN_TEXT = "#eaf1fb", "#6b93d6", "#2f5fa8"
LSTM_FILL, LSTM_STROKE, LSTM_CELL = "#fdf3e4", "#d39b4a", "#f7deb8"
FC_FILL, FC_STROKE, FC_LINK = "#e9f5ee", "#5aa57a", "#f1f8f3"
OUT_BAR, OUT_HI = "#d0d7de", "#24292f"
FIG_MUTED = "#4b5563"  # label details; darker than the page gray because CM strokes are thin

# Block labels: bold title, then detail lines (LaTeX).
LABELS = [
    ("I/Q frame", [r"$\mathbf{x} \in \mathbb{C}^{1024}$"]),
    ("Windowing", [r"$8 \times 224$ samples", r"50\% overlap"]),
    ("Input image", [r"$3 \times 224 \times 224$", r"I/Q, $A$, $\phi$"]),
    ("Modified AlexNet", [r"5 conv + GAP $\to \mathbb{R}^{256}$", r"shared weights"]),
    ("LSTM", [r"8 steps", r"hidden size 256"]),
    ("Fully connected", [r"$2048 \to 512 \to 256$"]),
    ("Prediction", [r"softmax, 9 classes"]),
]
EQUATION = (r"$\displaystyle A[n] = \sqrt{I[n]^2 + Q[n]^2}, \qquad "
            r"\phi[n] = \operatorname{atan2}\bigl(Q[n],\, I[n]\bigr)$")


def f(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- LaTeX text

def typeset(snippets):
    """Typeset each snippet on its own page; return [(viewBox, body)] plus shared glyph defs."""
    pages = "\n".join(r"\begin{lbl}" + s + r"\end{lbl}" for s in snippets)
    tex = (r"\documentclass[multi={lbl},border=0pt]{standalone}"
           r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}"
           r"\usepackage{xcolor}\definecolor{muted}{HTML}{" + FIG_MUTED[1:] + "}"
           r"\newenvironment{lbl}{}{}\begin{document}" + pages + r"\end{document}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "t.tex").write_text(tex)
        subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                       cwd=tmp, check=True, capture_output=True)
        subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "--precision=3", "--page=1-",
                        "-o", "p-%p.svg", "t.dvi"], cwd=tmp, check=True, capture_output=True)
        files = sorted(Path(tmp).glob("p-*.svg"), key=lambda p: int(p.stem.split("-")[1]))
        raw = [p.read_text() for p in files]

    glyphs, by_shape, out = {}, {}, []
    for k, svg in enumerate(raw):
        m = re.search(r"viewBox='([^']+)'", svg)
        if m is None:
            raise RuntimeError(f"dvisvgm page {k + 1} has no viewBox")
        vb = tuple(map(float, m.group(1).split()))
        rename = {}
        for gid, d in re.findall(r"<path id='([^']+)' d='([^']+)'/>", svg):
            if d not in by_shape:
                by_shape[d] = f"g{len(by_shape)}"
                glyphs[by_shape[d]] = d
            rename[gid] = by_shape[d]
        body = svg.split("</defs>", 1)[1].rsplit("</svg>", 1)[0]
        body = re.sub(r"<g id='page\d+'>", "<g>", body)
        body = re.sub(r"xlink:href='#([^']+)'", lambda m: f"href='#{rename[m.group(1)]}'", body)
        out.append((vb, body.strip()))
    return out, glyphs


def place(label, s, x, y, anchor="middle", fill=INK, valign="top"):
    """Put a typeset label at scale s; (x, y) is its top-centre, top-left, or centre-left."""
    (vx, vy, vw, vh), body = label
    tx = x - s * vx - (s * vw / 2 if anchor == "middle" else 0)
    ty = y - s * vy - (s * vh / 2 if valign == "middle" else 0)
    return f'<g fill="{fill}" transform="translate({f(tx)},{f(ty)}) scale({f(s)})">{body}</g>'


# ---------------------------------------------------------------- shapes

def rect(x, y, w, h, fill, stroke=None, rx=0, sw=1.25):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    r = f' rx="{f(rx)}"' if rx else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{r} fill="{fill}"{s}/>'


def iq_frame(x0, cy, t):
    w, h = 116, 80
    out = [rect(x0, cy - h / 2, w, h, N_FILL, N_STROKE, rx=6)]
    rng = np.random.default_rng(5)
    nsym, sps = 14, 8
    levels = np.array([-3, -1, 1, 3]) / 3
    sym = rng.choice(levels, nsym) + 1j * rng.choice(levels, nsym)
    up = np.zeros(nsym * sps, complex)
    up[::sps] = sym
    tt = np.arange(-3 * sps, 3 * sps + 1) / sps
    x = np.convolve(up, np.sinc(tt) * np.hanning(tt.size), mode="same")
    i_part, q_part = np.real(x), np.imag(x)
    scale = max(np.abs(i_part).max(), np.abs(q_part).max())
    xs = np.linspace(x0 + 10, x0 + w - 10, x.size)
    for comp, col in ((i_part / scale, I_COL), (q_part / scale, Q_COL)):
        pts = " ".join(f"{f(a)},{f(cy - 27 * b)}" for a, b in zip(xs, comp))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
    return out


def windows(x0, cy, t):
    w = 120
    seg, step = w * 224 / 1024, w * 112 / 1024  # to scale: 224-sample windows, 50% overlap
    out = [rect(x0, cy - 42, w, 6, "#d0d7de", rx=3)]
    out += [rect(x0 + i * step, cy - 28 + i * 9, seg, 6, ARROW, rx=3) for i in range(8)]
    return out


def input_image(x0, cy, t):
    s = 56
    out = [rect(x0 + 16, cy - 36, s, s, "#eef1f4", N_STROKE, rx=3),
           rect(x0 + 8, cy - 28, s, s, "#e3e8ed", N_STROKE, rx=3)]
    bands = 0.5 + 0.5 * np.sin(np.linspace(0, 5.2, 14)) * np.cos(np.linspace(0, 2.1, 14))
    for k, b in enumerate(bands):
        g = int(0xc4 + (0xf1 - 0xc4) * b)
        out.append(rect(x0, cy - 20 + k * s / 14, s, s / 14 + 0.3, f"#{g:02x}{g + 4:02x}{g + 8:02x}"))
    out.append(rect(x0, cy - 20, s, s, "none", N_STROKE, rx=3))
    out.append(place(t["times8"], 1.35, x0 + 76, cy - 42, anchor="start", fill=MUTED))
    return out


def cnn(x0, cy, t):
    w = 108
    return [f'<polygon points="{f(x0)},{f(cy - 48)} {f(x0 + w)},{f(cy - 20)} {f(x0 + w)},{f(cy + 20)} '
            f'{f(x0)},{f(cy + 48)}" fill="{CNN_FILL}" stroke="{CNN_STROKE}" stroke-width="1.25" '
            'stroke-linejoin="round"/>',
            place(t["cnn"], 1.45, x0 + 48, cy, fill=CNN_TEXT, valign="middle")]


def lstm(x0, cy, t):
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


def dense(x0, cy, t):
    bars = [(x0, cy - 50, 100), (x0 + 30, cy - 31, 62), (x0 + 60, cy - 21, 42)]  # 2048 -> 512 -> 256
    out = [f'<polygon points="{f(xa + 13)},{f(ya)} {f(xb)},{f(yb)} {f(xb)},{f(yb + hb)} '
           f'{f(xa + 13)},{f(ya + ha)}" fill="{FC_LINK}"/>'
           for (xa, ya, ha), (xb, yb, hb) in zip(bars, bars[1:])]
    out += [rect(x, y, 13, h, FC_FILL, FC_STROKE, rx=3) for x, y, h in bars]
    return out


def prediction(x0, cy, t):
    lengths = [9, 6, 12, 15, 8, 17, 11, 62, 7]
    return [rect(x0, cy - 43 + k * 10, ln, 6, OUT_HI if ln == max(lengths) else OUT_BAR, rx=2)
            for k, ln in enumerate(lengths)]


# draw function, shape width, shape half-height
SHAPES = [(iq_frame, 116, 40), (windows, 120, 42), (input_image, 72, 36), (cnn, 108, 48),
          (lstm, 136, 28), (dense, 73, 50), (prediction, 62, 43)]


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{ARROW}" '
            'stroke-width="1.5" marker-end="url(#arrow)"/>')


# ---------------------------------------------------------------- layouts

def wide(t, s=1.5, margin=20, min_gap=28, label_gap=14):
    cy, el = 70, []
    x0 = w = c = lw = 0.0
    tallest = 0.0
    for k, (draw, bw, hh) in enumerate(SHAPES):
        label = t["centered"][k]
        blw = s * label[0][2]
        if k == 0:
            bx0 = margin + max(0, (blw - bw) / 2)
        else:
            bx0 = max(x0 + w + min_gap, c + lw / 2 + label_gap + blw / 2 - bw / 2)
            el.append(arrow(x0 + w + 6, cy, bx0 - 8, cy))
        x0, w, lw = bx0, bw, blw
        c = x0 + w / 2
        el += draw(x0, cy, t)
        el.append(place(label, s, c, cy + 62))
        tallest = max(tallest, s * label[0][3])
    width = max(x0 + w, c + lw / 2) + margin
    return width, cy + 62 + tallest + margin, el


def vertical(t, s=1.6, margin=20, gap=34):
    widest = max(w for _, w, _ in SHAPES)
    col, label_x = margin + widest / 2, margin + widest + 28
    el, y, rows = [], margin, []
    for k, (draw, bw, hh) in enumerate(SHAPES):
        cy = y + hh
        el += draw(col - bw / 2, cy, t)
        el.append(place(t["left"][k], s, label_x, cy, anchor="start", valign="middle"))
        rows.append((y, y + 2 * hh))
        y += 2 * hh + gap
    el += [arrow(col, b + 6, col, top - 8) for (_, b), (top, _) in zip(rows, rows[1:])]
    width = label_x + max(s * lab[0][2] for lab in t["left"]) + margin
    return width, y - gap + margin, el


def svg_doc(width, height, glyphs, body, title, arrows=True, display=1.0):
    used = sorted(set(re.findall(r"href='#(g\d+)'", body)), key=lambda g: int(g[1:]))
    defs = "\n".join(f"<path id='{g}' d='{glyphs[g]}'/>" for g in used)
    marker = (f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
              f'markerUnits="userSpaceOnUse" orient="auto">\n<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/>\n'
              f'</marker>\n') if arrows else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{f(width * display)}" height="{f(height * display)}" viewBox="0 0 {f(width)} {f(height)}" role="img" aria-labelledby="title">
<title id="title">{title}</title>
<defs>
{marker}{defs}
</defs>
<rect width="{f(width)}" height="{f(height)}" fill="#fff"/>
{body}
</svg>
"""


def main():
    def stack(title, lines, align):
        rows = [r"\textbf{" + title + "}"] + [r"\color{muted}" + ln for ln in lines]
        return r"\begin{tabular}{@{}" + align + r"@{}}" + r"\\[1pt]".join(rows[:2]) + \
            "".join(r"\\" + r for r in rows[2:]) + r"\end{tabular}"

    snippets = ([stack(ti, ls, "c") for ti, ls in LABELS] + [stack(ti, ls, "l") for ti, ls in LABELS]
                + [r"\textbf{CNN}", r"\color{muted}$\times 8$", EQUATION])
    pages, glyphs = typeset(snippets)
    n = len(LABELS)
    t = {"centered": pages[:n], "left": pages[n:2 * n], "cnn": pages[2 * n], "times8": pages[2 * n + 1]}

    title = ("CNN-LSTM architecture: I/Q frame, eight overlapping windows, 3-channel input image, "
             "modified AlexNet, LSTM, fully connected layers, prediction over nine classes")
    for name, layout in (("architecture.svg", wide), ("architecture-vertical.svg", vertical)):
        width, height, el = layout(t)
        (OUT / name).write_text(svg_doc(width, height, glyphs, "\n".join(el), title))
        print(f"{name}: {f(width)} x {f(height)}")

    (vx, vy, vw, vh), body = pages[-1]
    pad = 1.0
    eq = svg_doc(vw + 2 * pad, vh + 2 * pad, glyphs, place(pages[-1], 1, pad + vw / 2, pad),
                 "A[n] = sqrt(I[n]^2 + Q[n]^2), phi[n] = atan2(Q[n], I[n])", arrows=False, display=1.8)
    (OUT / "equation.svg").write_text(eq)
    print(f"equation.svg: {f(1.8 * (vw + 2 * pad))} x {f(1.8 * (vh + 2 * pad))} px")


if __name__ == "__main__":
    main()
