from typing import Any
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ielts_judge.Criteria import CRITERION_IDS, CRITERION_SOURCE_NAMES, format_score_schema
from ielts_judge.Paths import PATHS


def prompt_to_diploma_image(
    output_path: str = PATHS["results"]["prompt_image"],
    width: int = 1600,
    padding: int = 80,
    bg_color: str = "#f8f8f5ff",
    text_color: str = "#1e1e1e",
    accent_color: str = "#038f6aaf",
) -> None:
    """Render the IELTS prompt using Pillow's built-in scalable font.

    Args:
        output_path (str): Output path used by this operation.
        width (int): Width used by this operation.
        padding (int): Padding used by this operation.
        bg_color (str): Bg color used by this operation.
        text_color (str): Text color used by this operation.
        accent_color (str): Accent color used by this operation.

    Returns:
        None: This operation returns no value.
    """
    prompt = rf"""
### Instruction:
You are an IELTS writing evaluator.
Evaluate the essay and return ONLY JSON, no text explanation.
Scores range from 0 to 9 with step 0.5.

Return format:
{format_score_schema()}

### Text:
{{essay}}

### Answer:
""".strip()

    # ↑ увеличиваем шрифты
    title_font = ImageFont.load_default(size=38)
    body_size = 32
    body_font = ImageFont.load_default(size=body_size)

    # ↓ меньше строк = текст крупнее визуально
    wrapper = textwrap.TextWrapper(width=52)

    wrapped_lines = []

    for line in prompt.split("\n"):
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(wrapper.wrap(line))

    # ↑ чуть больше воздуха между строками
    line_height = body_size + 18

    height = padding * 2 + 70 + line_height * len(wrapped_lines)

    img = Image.new("RGB", (int(width), int(height)), bg_color)
    draw = ImageDraw.Draw(img)

    # Заголовок
    title = "Zero-shot prompt"

    draw.text((padding, padding), title, fill=accent_color, font=title_font)

    # Линия
    y_line = padding + 68

    draw.line(
        [(padding, y_line), (width - padding, y_line)], fill=accent_color, width=4
    )
    text_height = line_height * len(wrapped_lines)
    current_target = y_line + 30

    # ↓ более плотная обводка вокруг текста
    box_padding_x = 14
    box_padding_y = 10

    box_top = current_target - box_padding_y
    box_bottom = current_target + text_height + box_padding_y

    draw.rounded_rectangle(
        [
            (padding - box_padding_x, box_top),
            (width - padding + box_padding_x, box_bottom),
        ],
        radius=16,  # меньше скругление = более “доковый” вид
        fill="#ffffff",
        outline="#cbd5e1",
        width=2,
    )

    # Текст
    for line in wrapped_lines:

        if line.startswith("###"):
            color = accent_color
        else:
            color = text_color

        draw.text((padding, current_target), line, fill=color, font=body_font)

        current_target += line_height

    # 300 dpi для диплома/печати
    img.save(output_path, dpi=(300, 300))

    print(f"Saved to {output_path}")


prompt_to_diploma_image()


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_aes_architecture(
    save_path: str = PATHS["results"]["aes_architecture_image"],
    num_heads: int = 4,
) -> None:
    """Clean thesis-ready AES architecture diagram (horizontal layout).
    Fixed arrows, full visibility, scalable heads.

    Args:
        save_path (str): Save path used by this operation.
        num_heads (int): Num heads used by this operation.

    Returns:
        None: This operation returns no value.
    """

    _, ax = plt.subplots(figsize=(20, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(sample: Any, current_target: Any, text: str, width: float=2.0, height: float=1.2, fc: str="#F5F5F5") -> dict[str, Any]:
        """Box.

        Args:
            sample (Any): Sample used by this operation.
            current_target (Any): Current target used by this operation.
            text (str): Text used by this operation.
            width (float): Width used by this operation.
            height (float): Height used by this operation.
            fc (str): Fc used by this operation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        rect = FancyBboxPatch(
            (sample, current_target),
            width,
            height,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="black",
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(
            sample + width / 2, current_target + height / 2, text, ha="center", va="center", fontsize=10
        )
        return {
            "left": (sample, current_target + height / 2),
            "right": (sample + width, current_target + height / 2),
            "center": (sample + width / 2, current_target + height / 2),
        }

    def arrow(p1: Any, p2: Any) -> None:
        """Arrow.

        Args:
            p1 (Any): P1 used by this operation.
            p2 (Any): P2 used by this operation.

        Returns:
            None: This operation returns no value.
        """
        ax.add_patch(
            FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=15, linewidth=1.5)
        )

    # =====================
    # MAIN PIPELINE
    # =====================
    b_input = box(0.3, 2.5, "Essay Input")
    b_enc = box(2.6, 2.5, "RoBERTa Encoder", fc="#DDEEFF")
    b_pool = box(5.0, 2.5, "Masked Mean\nPooling", fc="#E8F5E9")
    b_emb = box(7.4, 2.5, "Sentence\nEmbedding", fc="#FFF3E0")

    # =====================
    # HEADS (dynamic)
    # =====================
    heads = []
    y_positions = [4.5, 3.2, 2.0, 0.8]
    criteria = [CRITERION_SOURCE_NAMES[criterion_id] for criterion_id in CRITERION_IDS]
    for current_index in range(num_heads):
        heads.append(
            box(9.8, y_positions[current_index], f"Head {current_index+1}\n({criteria[current_index]})", fc="#FCE4EC")
        )

    # =====================
    # MAIN FLOW ARROWS (FIXED)
    # =====================

    # Input → Encoder
    arrow(b_input["right"], b_enc["left"])

    # Encoder → Pooling
    arrow(b_enc["right"], b_pool["left"])

    # Pooling → Embedding
    arrow(b_pool["right"], b_emb["left"])

    # =====================
    # BRANCHING (fixed to avoid overlap)
    # =====================
    for current_index, current_digest in enumerate(heads):
        arrow(b_emb["right"], (9.8, y_positions[current_index] + 0.6))

    # =====================
    # OUTPUT LABEL
    # =====================
    ax.text(
        10.8, 0.2, "Ordinal outputs\n(K-1 logits → score)", ha="center", fontsize=10
    )

    # =====================
    # TITLE
    # =====================
    ax.text(
        6,
        5.6,
        "AES Expert Model Architecture (Multi-task Ordinal Regression)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # =====================
    # FIX: prevent clipping on right side
    # =====================
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="webp")
