import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python launcher.py <input_path>")
    sys.exit(1)

input_path = sys.argv[1] 

scripts = [
    ("detect_shelves.py", True),   # gets the input path as an argument
    ("get_detected_shelves.py", False), # not critical just for merging the results of shelf detection
    ("get_masked_images.py", False),
    ("detect_products.py", False),
    ("merge_masked_detection_results.py", False),
    ("visualize_the_final_results.py", False),
]

for script, needs_input in scripts:
    print(f"{script} running...")

    cmd = [sys.executable, script]
    if needs_input:
        cmd.append(input_path)

    subprocess.run(cmd, check=True)

print("All processes completed.")