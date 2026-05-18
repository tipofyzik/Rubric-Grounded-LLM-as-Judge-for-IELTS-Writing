from PIL import Image, ImageDraw, ImageFont
import textwrap


def prompt_to_diploma_image(
    output_path="prompt.png",
    width=1600,
    padding=80,
    bg_color="#f8f8f5ff",
    text_color="#1e1e1e",
    accent_color="#038f6aaf",
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
):
    prompt = r"""
### Instruction:
You are an IELTS writing evaluator. 
Evaluate the essay and return ONLY JSON, no text explanation.
Scores range from 0 to 9 with step 0.5.

Return format:
{
    "task_achievement": number,
    "coherence_and_cohesion": number,
    "lexical_resource": number,
    "grammatical_range_and_accuracy": number
}

### Text:
{essay}

### Answer:
""".strip()

    # ↑ увеличиваем шрифты
    title_font = ImageFont.truetype(font_path, 38)
    body_font = ImageFont.truetype(font_path, 32)

    # ↓ меньше строк = текст крупнее визуально
    wrapper = textwrap.TextWrapper(width=52)

    wrapped_lines = []

    for line in prompt.split("\n"):
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(wrapper.wrap(line))

    # ↑ чуть больше воздуха между строками
    line_height = body_font.size + 18

    height = (
        padding * 2
        + 70
        + line_height * len(wrapped_lines)
    )

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Заголовок
    title = "Zero-shot Prompt for IELTS Essay Evaluation"

    draw.text(
        (padding, padding),
        title,
        fill=accent_color,
        font=title_font
    )

    # Линия
    y_line = padding + 68

    draw.line(
        [(padding, y_line), (width - padding, y_line)],
        fill=accent_color,
        width=4
    )
    text_height = line_height * len(wrapped_lines)
    y = y_line + 30

    # ↓ более плотная обводка вокруг текста
    box_padding_x = 14
    box_padding_y = 10

    box_top = y - box_padding_y
    box_bottom = y + text_height + box_padding_y

    draw.rounded_rectangle(
        [
            (padding - box_padding_x, box_top),
            (width - padding + box_padding_x, box_bottom)
        ],
        radius=16,          # меньше скругление = более “доковый” вид
        fill="#ffffff",
        outline="#cbd5e1",
        width=2
    )

    # Текст
    for line in wrapped_lines:

        if line.startswith("###"):
            color = accent_color
        else:
            color = text_color

        draw.text(
            (padding, y),
            line,
            fill=color,
            font=body_font
        )

        y += line_height

    # 300 dpi для диплома/печати
    img.save(output_path, dpi=(300, 300))

    print(f"Saved to {output_path}")


# prompt_to_diploma_image("./ielts_prompt.png")


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_aes_architecture(save_path: str = None, num_heads: int = 4):
    """
    Clean thesis-ready AES architecture diagram (horizontal layout).
    Fixed arrows, full visibility, scalable heads.
    """

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, text, width=2.0, height=1.2, fc="#F5F5F5"):
        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="black",
            facecolor=fc
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10
        )
        return {
            "left": (x, y + height / 2),
            "right": (x + width, y + height / 2),
            "center": (x + width / 2, y + height / 2)
        }

    def arrow(p1, p2):
        ax.add_patch(FancyArrowPatch(
            p1, p2,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.5
        ))

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
    criteria = ["task_achievement", 
                "coherence_and_cohesion", 
                "lexical_resource", 
                "grammatical_range_and_accuracy"]
    for i, criterion in enumerate(criteria):
        criteria[i] = criterion.replace("_", " ").title()
    for i in range(num_heads):
        heads.append(
            box(
                9.8,
                y_positions[i],
                f"Head {i+1}\n({criteria[i]})",
                fc="#FCE4EC"
            )
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
    for i, h in enumerate(heads):
        arrow(
            b_emb["right"],
            (9.8, y_positions[i] + 0.6)
        )

    # =====================
    # OUTPUT LABEL
    # =====================
    ax.text(
        10.8, 0.2,
        "Ordinal outputs\n(K-1 logits → score)",
        ha="center",
        fontsize=10
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
        fontweight="bold"
    )

    # =====================
    # FIX: prevent clipping on right side
    # =====================
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="webp")

# draw_aes_architecture("./aes_model.webp")