from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/albertorodriguezsalgado/Desktop/LLMs_mazes")
ASSETS = ROOT / "assets"
STRIP_PATH = ASSETS / "maze_strip_1_2_8_10.png"
PLOT_OUT = ASSETS / "benchmark_plot_ranked_with_latency_fair_v3.png"
STACK_OUT = ASSETS / "maze_strip_with_results_fair_v3.png"


MODELS = [
    {"label": "GPT-5.4 (medium)", "vendor": "OpenAI", "solved": 9, "latency": 51.06, "color": "#3d7a9e"},
    {"label": "GPT-5.4 (high)", "vendor": "OpenAI", "solved": 9, "latency": 74.2, "color": "#2f6788"},
    {"label": "Gemini 3.1 Pro", "vendor": "Google", "solved": 8, "latency": 82.75, "color": "#a08b5b"},
    {"label": "Gemini 3 Flash", "vendor": "Google", "solved": 6, "latency": 76.18, "color": "#a08b5b"},
    {"label": "Claude Opus 4.6", "vendor": "Anthropic", "solved": 2, "latency": None, "color": "#c94a3a"},
    {"label": "Qwen 3.5 Flash", "vendor": "Alibaba / Qwen", "solved": 1, "latency": 51.85, "color": "#2f7f69"},
    {"label": "GPT-5.1", "vendor": "OpenAI", "solved": 1, "latency": 85.39, "color": "#3d7a9e"},
    {"label": "GPT-5.4 (none)", "vendor": "OpenAI", "solved": 0, "latency": 4.24, "color": "#7aa8c4"},
    {"label": "Claude Sonnet 4.6", "vendor": "Anthropic", "solved": 0, "latency": None, "color": "#c94a3a"},
    {"label": "Claude Haiku 4.5", "vendor": "Anthropic", "solved": 0, "latency": 17.51, "color": "#c94a3a"},
    {"label": "Qwen 3.5 Plus", "vendor": "Alibaba / Qwen", "solved": 0, "latency": 67.95, "color": "#2f7f69"},
]


BG = "#f6f1e8"
TEXT = "#201b18"
MUTED = "#857a6e"
LINE = "#d7cdbe"
TRACK = "#ebe4d8"
N_A = "#cbc0b2"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    else:
        path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size=size)


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, bold: bool = False):
    size = start_size
    while size > 18:
        fnt = font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=fnt)
        if bbox[2] - bbox[0] <= max_width:
            return fnt
        size -= 1
    return font(18, bold=bold)


def fmt_latency(value):
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


def build_plot() -> Image.Image:
    width = 3300
    padding_x = 90
    top_pad = 110
    header_gap = 90
    row_h = 150
    bottom_pad = 110
    plot_h = top_pad + header_gap + row_h * len(MODELS) + bottom_pad

    image = Image.new("RGB", (width, plot_h), BG)
    draw = ImageDraw.Draw(image)

    title_font = font(56, bold=True)
    subtitle_font = font(26)
    header_font = font(22, bold=True)
    rank_font = font(30, bold=True)
    value_font = font(30, bold=True)
    latency_font = font(24, bold=True)
    vendor_font = font(24)

    inner_w = width - padding_x * 2
    left_w = 690
    bar_w = 1450
    score_w = 120
    latency_w = inner_w - left_w - bar_w - score_w - 40
    rank_r = 34
    bar_x = padding_x + left_w
    bar_h = 72
    score_x = bar_x + bar_w + 42
    latency_x = score_x + score_w + 40

    draw.text((padding_x, 26), "Exact shortest-path solves by model", fill=TEXT, font=title_font)
    draw.text(
        (padding_x, 80),
        "10 mazes • exact shortest path required • average thinking time shown at right",
        fill=MUTED,
        font=subtitle_font,
    )

    header_y = top_pad
    draw.text((padding_x + 120, header_y), "Model", fill=MUTED, font=header_font)
    draw.text((bar_x, header_y), "0", fill=MUTED, font=header_font)
    right_label = "10 solved"
    right_bbox = draw.textbbox((0, 0), right_label, font=header_font)
    draw.text((bar_x + bar_w - (right_bbox[2] - right_bbox[0]), header_y), right_label, fill=MUTED, font=header_font)
    latency_label = "Avg thinking time"
    latency_bbox = draw.textbbox((0, 0), latency_label, font=header_font)
    draw.text((latency_x + latency_w - (latency_bbox[2] - latency_bbox[0]), header_y), latency_label, fill=MUTED, font=header_font)
    draw.line((padding_x, header_y + 46, width - padding_x, header_y + 46), fill=LINE, width=2)

    for idx, model in enumerate(MODELS, start=1):
        row_top = top_pad + header_gap + (idx - 1) * row_h
        row_mid = row_top + row_h / 2
        row_bottom = row_top + row_h

        draw.line((padding_x, row_bottom, width - padding_x, row_bottom), fill=LINE, width=2)

        circle_x = padding_x + 38
        circle_y = row_mid - rank_r
        draw.ellipse((circle_x, circle_y, circle_x + rank_r * 2, circle_y + rank_r * 2), fill=model["color"])
        num_bbox = draw.textbbox((0, 0), str(idx), font=rank_font)
        num_w = num_bbox[2] - num_bbox[0]
        num_h = num_bbox[3] - num_bbox[1]
        draw.text(
            (circle_x + rank_r - num_w / 2, row_mid - num_h / 2 - 4),
            str(idx),
            fill=WHITE,
            font=rank_font,
        )

        label_x = padding_x + 118
        label_font = fit_text(draw, model["label"], left_w - 160, 38, bold=True)
        draw.text((label_x, row_top + 28), model["label"], fill=TEXT, font=label_font)
        draw.text((label_x, row_top + 78), model["vendor"], fill=MUTED, font=vendor_font)

        bar_y = row_mid - bar_h / 2
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=20, fill=TRACK)
        fill_w = max(8, int((model["solved"] / 10) * bar_w)) if model["solved"] > 0 else 0
        if fill_w:
            draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=20, fill=model["color"])

        score_text = str(model["solved"])
        score_bbox = draw.textbbox((0, 0), score_text, font=value_font)
        score_w_px = score_bbox[2] - score_bbox[0]
        score_h_px = score_bbox[3] - score_bbox[1]
        draw.text(
            (score_x + (score_w - score_w_px) / 2, row_mid - score_h_px / 2 - 4),
            score_text,
            fill=TEXT,
            font=value_font,
        )

        latency_text = fmt_latency(model["latency"])
        pill_fill = TEXT if model["latency"] is not None else N_A
        pill_text = WHITE if model["latency"] is not None else TEXT
        pill_w = latency_w - 10
        pill_h = 58
        pill_y = row_mid - pill_h / 2
        draw.rounded_rectangle((latency_x, pill_y, latency_x + pill_w, pill_y + pill_h), radius=18, fill=pill_fill)
        lat_bbox = draw.textbbox((0, 0), latency_text, font=latency_font)
        lat_w = lat_bbox[2] - lat_bbox[0]
        lat_h = lat_bbox[3] - lat_bbox[1]
        draw.text(
            (latency_x + pill_w / 2 - lat_w / 2, row_mid - lat_h / 2 - 3),
            latency_text,
            fill=pill_text,
            font=latency_font,
        )

    footer = "n/a indicates no clean full-run average latency was available."
    draw.text((padding_x, plot_h - 56), footer, fill=MUTED, font=font(22))
    return image


def build_stack(plot_image: Image.Image) -> Image.Image:
    width = plot_image.width
    bg = Image.new("RGB", (width, 1), BG)

    strip = Image.open(STRIP_PATH).convert("RGB")
    inner_w = width - 180
    strip_ratio = strip.height / strip.width
    strip_h = int(inner_w * strip_ratio)
    strip = strip.resize((inner_w, strip_h), Image.Resampling.LANCZOS)

    top_pad = 80
    gap = 70
    bottom_pad = 90
    total_h = top_pad + strip_h + gap + plot_image.height + bottom_pad
    canvas = bg.resize((width, total_h))
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle((90, top_pad, 90 + inner_w, top_pad + strip_h), radius=28, fill=WHITE)
    canvas.paste(strip, (90, top_pad))
    canvas.paste(plot_image, (0, top_pad + strip_h + gap))
    return canvas


def main():
    plot_image = build_plot()
    plot_image.save(PLOT_OUT, optimize=True)
    stack_image = build_stack(plot_image)
    stack_image.save(STACK_OUT, optimize=True)
    print(PLOT_OUT)
    print(STACK_OUT)


if __name__ == "__main__":
    main()
