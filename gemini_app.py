"""
Image Tagger with Arrow Annotations (Gemini + LangChain)

Upload an image and Gemini will find the objects (even small ones),
tag them and we draw an arrow pointing to each one on the image.

How it works (in 5 steps):
1. The user uploads an image and we convert it to base64 text.
2. A prompt template puts the image + instructions together.
3. A chain (prompt -> model -> parser) sends it to Gemini.
4. The parser turns Gemini's JSON reply into a Python list of objects,
   each with a label and a point (the centre of the object).
5. We draw an arrow and label for each object using Pillow.
"""

import base64
import io
import math
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

# ---------- Parser ----------
# Turns Gemini's JSON text into a normal Python list of dictionaries
parser = JsonOutputParser()

# ---------- Prompt ----------
# Note: curly braces that are part of the JSON example are doubled {{ }}
# so LangChain doesn't confuse them with placeholders like {image_data}.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            [
                {
                    "type": "text",
                    "text": """
Look very carefully at this image and detect EVERY distinct object you can see.
Do not skip small objects. Small items such as sunglasses, watches, jewellery,
phones, bottles, keys, bags, stickers, hats and shoes are just as important
as big objects. Also include objects in the background.

Reply ONLY with a JSON list. Each item must look like this:
{{"label": "sunglasses", "point": [y, x]}}

Rules:
- "label" is a short, simple, lowercase name (for example: laptop, van, backpack).
- "point" is the centre of that object, as 2 numbers between 0 and 1000
  relative to the image size: [y, x]. [0, 0] is the top-left corner.
- If the same kind of object appears several times, give each one its own item.
""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
    ]
)

# ---------- Model ----------
# Gemini is used through OpenRouter, so we use ChatOpenAI with a different base_url.
# temperature=0 makes the answers more consistent between runs.
llm = ChatOpenAI(
    model="google/gemini-3.7-flash",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_tokens=2000,
    temperature=0,
)

# ---------- Chain ----------
chain = prompt | llm | parser

# Colours for the arrows (they repeat if there are more objects)
COLORS = ["red", "blue", "green", "orange", "purple", "magenta", "cyan", "gold"]

# Each arrow starts from a different direction so labels don't pile up:
# top-left, top-right, bottom-left, bottom-right
DIRECTIONS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]


def draw_arrows(image, detections):
    """Draw a coloured arrow and label for every detected object."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Sizes depend on the image size so they look right on big and small images
    font = ImageFont.load_default(size=max(18, width // 45))
    arrow_length = width // 10
    line_width = max(3, width // 300)
    head_size = arrow_length // 3

    for i, obj in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        y, x = obj["point"]

        # Gemini gives numbers from 0 to 1000, so we convert them to real pixels.
        # This is where the arrow points to (the tip).
        tip_x = x / 1000 * width
        tip_y = y / 1000 * height

        # This is where the arrow starts (and where the label goes).
        # We keep it inside the image.
        step_x, step_y = DIRECTIONS[i % 4]
        start_x = min(max(tip_x + step_x * arrow_length, width * 0.1), width * 0.9)
        start_y = min(max(tip_y + step_y * arrow_length, height * 0.05), height * 0.95)

        # 1. The arrow line
        draw.line([(start_x, start_y), (tip_x, tip_y)], fill=color, width=line_width)

        # 2. The arrow head (a small triangle at the tip)
        angle = math.atan2(tip_y - start_y, tip_x - start_x)
        left_point = (
            tip_x - head_size * math.cos(angle - 0.5),
            tip_y - head_size * math.sin(angle - 0.5),
        )
        right_point = (
            tip_x - head_size * math.cos(angle + 0.5),
            tip_y - head_size * math.sin(angle + 0.5),
        )
        draw.polygon([(tip_x, tip_y), left_point, right_point], fill=color)

        # 3. The label, on a coloured background, at the start of the arrow
        text_box = draw.textbbox((start_x, start_y), obj["label"], font=font, anchor="mm")
        draw.rectangle(text_box, fill=color)
        draw.text((start_x, start_y), obj["label"], fill="white", font=font, anchor="mm")

    return annotated


# ---------- Streamlit page ----------
st.set_page_config(page_title="Image Tagger (Gemini)", page_icon="🏷️", layout="wide")
st.title("🏷️ Image Tagger (Gemini)")
st.write("Upload a photo and Gemini AI will tag and point to the objects it finds.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Only shrink very large images. A bigger image keeps more detail,
    # which helps Gemini notice small objects like sunglasses.
    image.thumbnail((2048, 2048))

    # Convert the image to base64 text so it can be sent inside the prompt
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with st.spinner("Gemini is looking at your image..."):
        detections = chain.invoke({"image_data": image_data})

    annotated_image = draw_arrows(image, detections)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Annotated")
        st.image(annotated_image, use_container_width=True)

    # ---------- Tags ----------
    st.subheader("Tags")
    st.markdown(" ".join(f":blue-badge[{obj['label']}]" for obj in detections))
    st.write(f"Total objects found: {len(detections)}")
else:
    st.info("👆 Upload a JPG or PNG image to get started.")
