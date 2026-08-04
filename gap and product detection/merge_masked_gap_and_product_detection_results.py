import os
import json
from pathlib import Path

VIDEO_NAME = Path("config.txt").read_text(encoding="utf-8").strip()

PRODUCT_ROOT = fr"product_detection_results\{VIDEO_NAME}"
GAP_ROOT = fr"gap_detection_results\{VIDEO_NAME}"

OUTPUT_ROOT = fr"merged_detection_results\{VIDEO_NAME}"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

frame_dirs = sorted([
    d for d in os.listdir(PRODUCT_ROOT)
    if os.path.isdir(os.path.join(PRODUCT_ROOT, d))
])

for frame_name in frame_dirs:

    print(f"Merging {frame_name}")

    product_frame_dir = os.path.join(PRODUCT_ROOT, frame_name)
    gap_frame_dir = os.path.join(GAP_ROOT, frame_name)

    save_dir = os.path.join(OUTPUT_ROOT, frame_name)
    os.makedirs(save_dir, exist_ok=True)

    merged = {
        "frame_name": frame_name,
        "shelves": []
    }

    product_files = sorted([
        f for f in os.listdir(product_frame_dir)
        if f.endswith(".json")
        and f != "merged_result.json"
    ])

    for product_file in product_files:

        shelf_id = int(
            os.path.splitext(product_file)[0].split("_")[-1]
        )

        product_path = os.path.join(
            product_frame_dir,
            product_file
        )

        gap_path = os.path.join(
            gap_frame_dir,
            f"shelf_{shelf_id}.json"
        )

        with open(product_path, "r", encoding="utf-8") as f:
            product_json = json.load(f)

        if os.path.exists(gap_path):

            with open(gap_path, "r", encoding="utf-8") as f:
                gap_json = json.load(f)

            shelf_result = {
                "shelf_id": shelf_id,
                "products": product_json.get("products", []),
                "category_summary": product_json.get("category_summary", []),
                "gaps": gap_json.get("gaps", []),
                "gap_summary": gap_json.get("gap_summary", {})
            }

        else:

            print(f"Gap result not found for shelf {shelf_id}")

            shelf_result = {
                "shelf_id": shelf_id,
                "products": product_json.get("products", []),
                "category_summary": product_json.get("category_summary", []),
                "gaps": [],
                "gap_summary": {}
            }

        merged["shelves"].append(shelf_result)

    save_path = os.path.join(
        save_dir,
        "merged_result.json"
    )

    with open(save_path, "w", encoding="utf-8") as f:

        json.dump(
            merged,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {save_path}")

print("\nDone.")