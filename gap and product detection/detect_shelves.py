"""
File:
    Shelf Region Analyzer

Purpose:
    This script analyzes retail shelf images or videos to detect Product Storage
    Regions using Google's Gemini 3.5 Flash Lite vision-language model.
    For each visible shelf storage region, it estimates occupancy, emptiness,
    confidence, and returns polygon annotations in JSON format.

Input:
    - An image file or a video file provided as a command-line argument

Output:
    - A results directory containing:
        - Original image (for image inputs)
        - Sampled video frames (optional)
        - JSON files containing the detected shelf regions and their attributes

Main Functionality:
    - Reads images and videos using OpenCV.
    - Samples video frames at fixed time intervals.
    - Sends each frame to the Gemini model together with a structured prompt.
    - Receives structured JSON containing shelf region information.
    - Saves the analysis results for further processing.

Model:
    - Google Gemini 3.5 Flash Lite
      (gemini-3.5-flash-lite)

Model Output:
    - is_shelving_present
    - shelf_regions:
        - region_id
        - polygon
        - occupancy_percentage
        - is_empty
        - confidence
        """
        
import sys
from time import sleep
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

from dotenv import load_dotenv
# =====================================================
# PARAMETERS
# =====================================================

load_dotenv()

INPUT_PATHS = sys.argv[1:]


SAMPLE_INTERVAL_SEC = 3
RETURN_POLYGONS = True
SAVE_FRAMES = True

RESULTS_ROOT = "shelf_detection_results"   


PROMPT = """
You are analyzing a retail store shelf.

Your task is NOT to detect shelf boards.

Instead, detect every Product Storage Region.

Definition:

A Product Storage Region is the complete usable storage space allocated to a single shelf level.

The region represents the empty volume where products can be placed.

Imagine every product has been removed before drawing the region.

The annotation should include the complete visible storage opening.

The region is bounded by:

- the shelf board above
- the shelf board below
- left/right dividers or visible shelf edges

Ignore:

- products
- shelf boards
- rack frames
- price tags
- labels
- advertisements
- shopping carts
- customers

If part of the storage region is occluded by products, infer the hidden boundaries.

Return one polygon for every visible Product Storage Region.

Polygon requirements:

- normalized coordinates in [0,1000]
- each point is [y,x]
- clockwise order
- 4 to 12 points
- use straight edges whenever possible

For every region estimate:

- occupancy_percentage (0-100)
- is_empty
- confidence (0-1)

Return ONLY valid JSON.
"""

if RETURN_POLYGONS:
    PROMPT += """

        For every shelf region return one bounding box tightly enclosing
        the usable storage area.

        The bounding box should cover the complete interior product storage
        space instead of the shelf board itself.

        Coordinates must be normalized to 0-1000 as

        [ymin, xmin, ymax, xmax]
        """

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

client = genai.Client(api_key=os.getenv("API_KEY_OTHER_SERVICE"))

# =====================================================
# GEMINI
# =====================================================

schema = {
    "type": "OBJECT",
    "properties": {

        "is_shelving_present": {
            "type": "BOOLEAN"
        },

        "shelf_regions": {

            "type": "ARRAY",

            "items": {

                "type": "OBJECT",

                "properties": {

                    "region_id": {
                        "type": "INTEGER"
                    },

                    "polygon": {

                        "type": "ARRAY",

                        "items": {

                            "type": "ARRAY",

                            "items": {
                                "type": "INTEGER"
                            },

                            "minItems": 2,
                            "maxItems": 2

                        }

                    },

                    "occupancy_percentage": {
                        "type": "INTEGER"
                    },

                    "is_empty": {
                        "type": "BOOLEAN"
                    },

                    "confidence": {
                        "type": "NUMBER"
                    }

                },

                "required": [
                    "region_id",
                    "polygon",
                    "occupancy_percentage",
                    "is_empty",
                    "confidence"
                ]

            }

        }

    },

    "required": [
        "is_shelving_present",
        "shelf_regions"
    ]
}


def analyze_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)
    image.thumbnail((640, 640))

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            PROMPT,
            image
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema
        )
    )

    return json.loads(response.text)

# =====================================================
# IMAGE
# =====================================================


def process_image(image_path):

    parent_folder = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    output_name = f"{parent_folder}_{image_name}"

    output_dir = os.path.join(
        RESULTS_ROOT,
        output_name
    )

    os.makedirs(output_dir, exist_ok=True)
    with open("config.txt", "w", encoding="utf-8") as f:
        f.write(Path(output_dir).name)
    # copy the original image to the output directory
    shutil.copy(image_path, output_dir)

    buffer = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    #frame = cv2.imread(image_path) # tr chracter problem

    if frame is None:
        print("Cannot read:", image_path)
        return

    result = analyze_frame(frame)
    sleep(5)
    with open(
        os.path.join(output_dir, image_name + ".json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(result, f, indent=4, ensure_ascii=False)

    print(image_name, "done.")

# =====================================================
# VIDEO
# =====================================================

def process_video(video_path):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.join(
        RESULTS_ROOT,
        f"{video_name}"
        f"_{SAMPLE_INTERVAL_SEC}s"
        f"_polygon"
        f"_frames-{int(SAVE_FRAMES)}"
    )

    output_dir = get_unique_dir(output_dir)

    
    with open("config.txt", "w", encoding="utf-8") as f:
        f.write(Path(output_dir).name)

    os.makedirs(output_dir)

    os.makedirs(output_dir, exist_ok=True)


    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:

            current_sec = frame_id / fps

            print(f"{video_name}  {current_sec:.1f}s")

            frame_name = f"{current_sec:.1f}s"

            if SAVE_FRAMES:

                cv2.imwrite(
                    os.path.join(output_dir, frame_name + ".jpg"),
                    frame
                )

            result = analyze_frame(frame)

            with open(
                os.path.join(output_dir, frame_name + ".json"),
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(result, f, indent=4, ensure_ascii=False)

        frame_id += 1

    cap.release()

# =====================================================
# INPUTS
# =====================================================

def collect_inputs(path):

    files = []

    if os.path.isfile(path):
        files.append(path)

    elif os.path.isdir(path):

        for file in sorted(os.listdir(path)):
            files.append(os.path.join(path,file))

    return files

# =====================================================
# UNIQUE DIRECTORY
# =====================================================

def get_unique_dir(path):
    if not os.path.exists(path):
        return path

    counter = 1
    while True:
        new_path = f"{path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# =====================================================
# MAIN
# =====================================================

os.makedirs(RESULTS_ROOT, exist_ok=True)

all_inputs = []

for path in INPUT_PATHS:
    all_inputs.extend(collect_inputs(path))

for file in all_inputs:

    ext = os.path.splitext(file)[1].lower() 

    try:

        if ext in IMAGE_EXTENSIONS:

            print("\nIMAGE:", file)
            process_image(file)

        elif ext in VIDEO_EXTENSIONS:

            print("\nVIDEO:", file)
            process_video(file)

        else:

            print("Skipped:", file)

    except Exception as e:

        print(file)
        print(e)

print("\nFinished.")