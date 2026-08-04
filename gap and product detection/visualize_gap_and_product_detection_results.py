import os
import json
from pathlib import Path
import cv2
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

VIDEO_NAME = Path("config.txt").read_text(encoding="utf-8").strip()

SHELF_RESULTS_FOLDER = fr"shelf_detection_results\{VIDEO_NAME}"

MERGED_RESULTS_FOLDER = (
    fr"merged_detection_results\{VIDEO_NAME}"
)

OUTPUT_FOLDER = (
    fr"final_visualizations\{VIDEO_NAME}"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# Utils
# --------------------------------------------------

def polygon_to_pixels(polygon, width, height):

    pts = []

    for point in polygon:

        y, x = point

        px = int(x / 1000 * width)
        py = int(y / 1000 * height)

        pts.append([px, py])

    return np.array([pts], dtype=np.int32)


def bbox_to_pixels(bbox, width, height):

    ymin, xmin, ymax, xmax = bbox

    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)

    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)

    return x1, y1, x2, y2


# --------------------------------------------------
# Process All Frames
# --------------------------------------------------

image_files = sorted([
    f for f in os.listdir(SHELF_RESULTS_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

font = cv2.FONT_HERSHEY_SIMPLEX

for image_file in image_files:

    frame_name = os.path.splitext(image_file)[0]

    print(f"Processing {frame_name}")

    image_path = os.path.join(
        SHELF_RESULTS_FOLDER,
        image_file
    )

    shelf_json_path = os.path.join(
        SHELF_RESULTS_FOLDER,
        frame_name + ".json"
    )

    merged_json_path = os.path.join(
        MERGED_RESULTS_FOLDER,
        frame_name,
        "merged_result.json"
    )

    if not os.path.exists(shelf_json_path):
        print(f"Shelf json not found: {frame_name}")
        continue

    if not os.path.exists(merged_json_path):
        print(f"Merged json not found: {frame_name}")
        continue

    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    with open(shelf_json_path, "r", encoding="utf-8") as f:
        shelf_json = json.load(f)

    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_json = json.load(f)

    # shelf_id -> has_gap
    gap_lookup = {}

    for shelf in merged_json["shelves"]:
        gap_lookup[shelf["shelf_id"]] = len(shelf.get("gaps", [])) > 0

    # --------------------------------------------------
    # Draw Shelf Polygons
    # --------------------------------------------------

    overlay = img.copy()

    for region in shelf_json["shelf_regions"]:

        pts = polygon_to_pixels(
            region["polygon"],
            w,
            h
        )

        has_gap = gap_lookup.get(region["region_id"], False)

        color = (
            (0, 0, 255)          # Red -> shelf has gap
            if has_gap
            else (255, 255, 255) # White -> no gap
        )

        cv2.fillPoly(
            overlay,
            pts,
            color
        )

    img = cv2.addWeighted(
        overlay,
        0.20,
        img,
        0.80,
        0
    )

    for region in shelf_json["shelf_regions"]:

        pts = polygon_to_pixels(
            region["polygon"],
            w,
            h
        )

        has_gap = gap_lookup.get(region["region_id"], False)

        color = (
            (0, 0, 255)
            if has_gap
            else (255, 255, 255)
        )

        cv2.polylines(
            img,
            pts,
            True,
            color,
            2
        )

    # --------------------------------------------------
    # Draw Products & Gaps
    # --------------------------------------------------

    category_counter = {}
    total_gap_count = 0
    gap_category_lines = []

    for shelf in merged_json["shelves"]:
        # ---------------- Summary for gaps ----------------
        if shelf.get("gaps"):

            categories = ", ".join(
                item["category"]
                for item in shelf.get("category_summary", [])
            )

            for gap in shelf["gaps"]:

                gap_category_lines.append(
                    f"Shelf {shelf['shelf_id']} - Gap {gap.get('gap_id', '')}: {categories}"
                )

        # ---------------- Products ----------------

        for product in shelf.get("products", []):

            x1, y1, x2, y2 = bbox_to_pixels(
                product["bbox"],
                w,
                h
            )

            category = product["category"]

            category_counter[category] = (
                category_counter.get(category, 0) + 1
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                product["product_name"],
                (x1, max(20, y1 - 18)),
                font,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                img,
                category,
                (x1, max(40, y1 - 3)),
                font,
                0.40,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ---------------- Gaps ----------------

        for gap in shelf.get("gaps", []):

            total_gap_count += 1


            x1, y1, x2, y2 = bbox_to_pixels(
                gap["bbox"],
                w,
                h
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),   # Blue
                2
            )

            cv2.putText(
                img,
                f"Gap {gap.get('gap_id', '')}",
                (x1, max(20, y1 - 8)),
                font,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    # --------------------------------------------------
    # Draw Category Summary
    # --------------------------------------------------

    lines = [
        f"Gaps: {total_gap_count}"
    ]

    lines.extend(
        f"{k}: {v}"
        for k, v in sorted(category_counter.items())
    )

    if lines:

        box_width = 260
        line_height = 22
        box_height = line_height * len(lines) + 20

        x0 = w - box_width - 10
        y0 = 10

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (0, 0, 0),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.60,
            img,
            0.40,
            0
        )

        y = y0 + 25

        for line in lines:

            cv2.putText(
                img,
                line,
                (x0 + 10, y),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            y += line_height


    # --------------------------------------------------
    # Draw Gap Category Summary (Bottom Right)
    # --------------------------------------------------

    if gap_category_lines:

        box_width = 300
        line_height = 22
        box_height = line_height * len(gap_category_lines) + 20

        x0 = w - box_width - 10
        y0 = h - box_height - 10

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (0, 0, 0),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.60,
            img,
            0.40,
            0
        )

        y = y0 + 25

        for line in gap_category_lines:

            cv2.putText(
                img,
                line,
                (x0 + 10, y),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            y += line_height
            
    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = os.path.join(
        OUTPUT_FOLDER,
        image_file
    )

    cv2.imwrite(output_path, img)

    print(f"Saved -> {output_path}")



print("\nDone.")