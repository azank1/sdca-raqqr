"""Generate docs/assets/quickstart.gif — animated terminal recording.

Captures real output from the quickstart, then renders it as a smooth
animated GIF using Pillow with a genuine dark-terminal appearance.

Run from the repo root:
    python docs/make_quickstart_gif.py
"""
import os
import sys
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ── colours (GitHub dark theme inspired) ─────────────────────────────────────
BG        = (13,  17,  23)     # #0d1117
FG        = (230, 237, 243)    # #e6edf3
PROMPT_C  = (63, 185,  80)    # #3fb950  green
CMD_C     = (88, 166, 255)    # #58a6ff  blue
HEADER_C  = (255, 166,  77)   # #ffa64d  orange (section headers)
NUM_C     = (121, 192, 255)   # #79c0ff  cyan (numbers in table)
DIM_C     = (139, 148, 158)   # #8b949e  grey

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE = 14
PAD_X, PAD_Y = 18, 14
LINE_H = 20

WIDTH  = 900
FRAME_HOLD = 4   # centiseconds per line (×2 at pauses)


def load_font(size=FONT_SIZE):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def line_colour(text: str) -> tuple:
    """Guess a colour from line content."""
    t = text.strip()
    if t.startswith("$"):
        return CMD_C
    if t.startswith("=== "):
        return HEADER_C
    for kw in ("days ", "buy_days", "sell_days", "no_trade", "starting_cash",
               "portfolio", "return_pct", "lump_", "vs_lump", "pnl", "avg_",
               "cash", "btc", "Loaded", "Fetching"):
        if kw in t:
            return NUM_C
    if t == "":
        return FG
    return FG


def collect_output() -> list[str]:
    """Run the quickstart via subprocess, collect lines."""
    script = textwrap.dedent("""\
        import sdca_core as sc
        print("Fetching BTC daily data from Binance...")
        ohlcv = sc.data.load_binance("BTCUSDT")
        print(f"Loaded {len(ohlcv)} days  "
              f"({ohlcv.index[0].date()} \\u2192 {ohlcv.index[-1].date()})")
        print()
        table = sc.analyze(ohlcv)
        cols = ["close","0.01","0.25","0.5","0.75","0.99","eqm_risk","composite_risk"]
        print("=== RAQQR bands + signals (last 5 rows) ===")
        print(table[cols].tail().to_string())
        print()
        res = sc.backtest_curve(ohlcv, starting_cash=10_000, start="2018-01-01")
        print("=== Backtest summary ===")
        print(res.summary().to_string())
    """)

    venv_py = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".venv", "bin", "python")
    py = venv_py if os.path.exists(venv_py) else sys.executable

    print("Running quickstart (this fetches live data)...")
    result = subprocess.run([py, "-c", script],
                            capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw = result.stdout.splitlines()
    print(f"  captured {len(raw)} lines")

    lines: list[str] = []
    lines.append("$ python examples/quickstart.py")
    lines.append("")
    for ln in raw:
        lines.append(ln)
    lines.append("")
    lines.append("$ _")
    return lines


def render_frame(lines_so_far: list[str], font: ImageFont.FreeTypeFont,
                 height: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    y = PAD_Y
    for ln in lines_so_far:
        colour = line_colour(ln)
        # prompt dollar sign gets its own colour
        if ln.startswith("$ "):
            draw.text((PAD_X, y), "$ ", font=font, fill=PROMPT_C)
            draw.text((PAD_X + int(font.getlength("$ ")), y),
                      ln[2:], font=font, fill=CMD_C)
        else:
            draw.text((PAD_X, y), ln, font=font, fill=colour)
        y += LINE_H
    return img


def make_gif(lines: list[str], out_path: str):
    font = load_font()
    height = PAD_Y * 2 + LINE_H * (len(lines) + 1)
    height = max(height, 400)

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Build up line-by-line
    for i, ln in enumerate(lines):
        shown = lines[:i + 1]
        frame = render_frame(shown, font, height)
        frames.append(frame)

        # longer pause on blank lines and section headers
        if ln.strip() == "" or ln.startswith("==="):
            durations.append(FRAME_HOLD * 3 * 10)
        elif ln.startswith("$ "):
            durations.append(FRAME_HOLD * 2 * 10)
        else:
            durations.append(FRAME_HOLD * 10)

    # Hold last frame for 4 s
    durations[-1] = 400

    # Convert to P mode (palette) for GIF
    palette_frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                      for f in frames]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    palette_frames[0].save(
        out_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Saved → {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    lines = collect_output()
    out = os.path.join(os.path.dirname(__file__), "assets", "quickstart.gif")
    make_gif(lines, out)
