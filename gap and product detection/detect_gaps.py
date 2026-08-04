import os
import json
from pathlib import Path
from PIL import Image
from time import sleep
from google import genai
from google.genai import types

from dotenv import load_dotenv
# ============================================
# PROMPT AND SCHEMA
# ============================================
GAP_PROMPT = """
You are an expert retail shelf auditing system.

The image contains only ONE shelf region.

Your task is to detect every missing product facing.

A missing facing is an empty shelf area where one or more products are expected to be placed but are currently missing.

Think like a supermarket employee checking shelves for restocking.

A valid missing facing should satisfy most of the following:

- It lies on the shelf surface.
- It represents a location where products are normally displayed.
- It is surrounded by products on one or both sides, or by products on one side and the shelf boundary on the other.
- Its depth is similar to the neighboring products.
- It is large enough to accommodate at least one product.

Do NOT detect:

- Normal spacing between adjacent products.
- Small gaps caused by product packaging.
- Tiny vertical or horizontal slits.
- Shadows.
- Reflections.
- Shelf labels.
- Price tags.
- Shelf rails.
- Shelf supports.
- Rack structures.
- Dark background visible between products.
- Empty pixels that are not actual missing facings.

If multiple missing positions belong to one continuous empty shelf region, return ONE bounding box covering the entire missing facing.

Do not split one missing facing into multiple boxes.

Bounding boxes must tightly surround only the missing facing.

Return normalized bounding boxes in the format:

[ymin, xmin, ymax, xmax]

All coordinates must be between 0 and 1000.

Also estimate the confidence of each detection.

Return JSON only.

"""

GAP_SCHEMA = {
    "type": "OBJECT",
    "required": [
        "gaps",
        "gap_summary"
    ],

    "properties": {

        "gaps": {
            "type": "ARRAY",

            "items": {
                "type": "OBJECT",

                "required": [
                    "gap_id",
                    "bbox"
                ],

                "properties": {

                    "gap_id": {
                        "type": "INTEGER"
                    },

                    "bbox": {
                        "type": "ARRAY",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "INTEGER"
                        }
                    },

                    "gap_size": {
                        "type": "STRING"
                    },

                    "confidence": {
                        "type": "NUMBER"
                    }
                }
            }
        },

        "gap_summary": {

            "type": "OBJECT",

            "properties": {

                "total_gaps": {
                    "type": "INTEGER"
                },

                "gap_percentage": {
                    "type": "NUMBER"
                }
            }
        }
    }
}


# ============================================
# CONFIG
# ============================================
VIDEO_NAME = Path("config.txt").read_text(encoding="utf-8").strip()
INPUT_ROOT = fr"masked_shelves\{VIDEO_NAME}"
OUTPUT_ROOT = "gap_detection_results"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

load_dotenv()

# ============================================
# GEMINI
# ============================================

client = genai.Client(api_key=os.getenv("API_KEY_GAP_DETECTION"))



def analyze_gaps(image_path):

    with Image.open(image_path) as image:

        response = client.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=[
                GAP_PROMPT,
                image
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json",

                response_schema=GAP_SCHEMA

            )
        )

    return json.loads(response.text)


# =====================================================
# MAIN
# =====================================================

input_folder_name = os.path.basename(os.path.normpath(INPUT_ROOT))

output_video_folder = os.path.join(
    OUTPUT_ROOT,
    input_folder_name
)

os.makedirs(output_video_folder, exist_ok=True)


frame_dirs = sorted([
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
])


for frame_name in frame_dirs:

    frame_dir = os.path.join(INPUT_ROOT, frame_name)

    save_dir = os.path.join(
        output_video_folder,
        frame_name
    )

    os.makedirs(save_dir, exist_ok=True)

    print(f"\nProcessing frame: {frame_name}")

    shelf_images = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for image_name in shelf_images:

        image_path = os.path.join(frame_dir, image_name)

        shelf_id = int(
            os.path.splitext(image_name)[0].split("_")[-1]
        )

        print(f"   Shelf {shelf_id}")

        sleep(5)

        result = analyze_gaps(image_path)

        result["shelf_id"] = shelf_id

        json_path = os.path.join(
            save_dir,
            f"shelf_{shelf_id}.json"
        )

        with open(json_path, "w", encoding="utf-8") as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )

print("\nDone.")