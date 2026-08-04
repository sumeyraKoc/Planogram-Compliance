import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python launcher.py <input_path>")
    sys.exit(1)

input_path = sys.argv[1]

# Sequential
subprocess.run(
    [sys.executable, "detect_shelves.py", input_path],
    check=True
)

subprocess.run(
    [sys.executable, "get_masked_images.py"],
    check=True
)

# Parallel
print("Starting product and gap detection...")

product_process = subprocess.Popen(
    [sys.executable, "detect_products.py"]
)

gap_process = subprocess.Popen(
    [sys.executable, "detect_gaps.py"]
)

product_process.wait()
gap_process.wait()

print("Product & gap detection completed.")

# Sequential again
subprocess.run(
    [sys.executable,
     "merge_masked_gap_and_product_detection_results.py"],
    check=True
)

subprocess.run(
    [sys.executable,
    "visualize_gap_and_product_detection_results.py"],
    check=True
)

print("\nAll processes completed.")