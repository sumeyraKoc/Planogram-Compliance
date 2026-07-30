import os
import json
import cv2
import numpy as np
from pathlib import Path


RESULTS_DIR = "shelf_detection_results"


def draw_json_text(img, result):

    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"Shelving Present : {result.get('is_shelving_present', False)}",
        f"Regions          : {len(result.get('shelf_regions', []))}"
    ]

    overlay = img.copy()
    
    box_height = 35 * len(lines) + 20

    cv2.rectangle(
        overlay,
        (10, 10),
        (430, box_height),
        (0, 0, 0),
        -1
    )

    img = cv2.addWeighted(
        overlay,
        0.6,
        img,
        0.4,
        0
    )

    y = 40

    for line in lines:

        cv2.putText(
            img,
            line,
            (20, y),
            font,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        y += 35

    img = draw_polygons(img, result)

    return img


def draw_polygons(img, result):

    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, region in enumerate(result.get("shelf_regions", [])):

        polygon = region.get("polygon", [])

        if len(polygon) < 3:
            continue

        pts = []

        for point in polygon:

            if len(point) != 2:
                continue

            y, x = point

            px = int(x / 1000 * w)
            py = int(y / 1000 * h)

            pts.append([px, py])

        if len(pts) < 3:
            continue

        pts = np.array([pts], dtype=np.int32)
        overlay = img.copy()

        cv2.fillPoly(
            overlay,
            pts,
            (0, 0, 255)
        )

        img = cv2.addWeighted(
            overlay,
            0.2,
            img,
            0.8,
            0
        )

        cv2.polylines(
            img,
            pts,
            True,
            (0, 0, 255),
            3
        )

        occ = region.get("occupancy_percentage", "?")
        conf = region.get("confidence", 0)

        label = (
            f"Region {region.get('region_id', i)} | "
            f"{occ}% | "
            f"{conf:.2f}"
        )

        M = cv2.moments(pts[0])

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pts[0][0]

        cv2.putText(
            img,
            label,
            (cx - 40, cy),
            font,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return img


# ---------------------------------------------------
# PROCESSING
# ---------------------------------------------------
folder_name = Path("config.txt").read_text(encoding="utf-8").strip()

folder_path = os.path.join(RESULTS_DIR, folder_name)


image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

json_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".json")
])

single_case = (len(image_files) == 1 and len(json_files) == 1)

if not single_case:
    merged_dir = os.path.join(folder_path, "merged_result")
    os.makedirs(merged_dir, exist_ok=True)

for image_file in image_files:

    image_path = os.path.join(folder_path, image_file)

    base = os.path.splitext(image_file)[0]

    json_path = os.path.join(folder_path, base + ".json")

    if not os.path.exists(json_path):
        json_path = os.path.join(folder_path, "result.json")

    if not os.path.exists(json_path):
        print(f"No json for {image_file}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    img = cv2.imread(image_path)

    img = draw_json_text(img, result)

    output_name = f"{base}_merged.jpg"

    if single_case:
        save_path = os.path.join(folder_path, output_name)
    else:
        save_path = os.path.join(merged_dir, output_name)

    cv2.imwrite(save_path, img)

    print("Saved:", save_path)

print("\nDone!")