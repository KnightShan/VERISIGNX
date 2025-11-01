from PIL import Image
import os
import numpy as np
import cv2
from pathlib import Path


def main():
    images_dir = "../OCR/OCR_Results"
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), images_dir)

    if not os.path.isdir(input_path):
        print("Input folder not found:", input_path)
        return

    ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    total_files = 0
    processed_files = 0

    for filename in sorted(os.listdir(input_path)):
        if Path(filename).suffix.lower() not in ALLOWED:
            continue

        total_files += 1
        file_path = os.path.join(input_path, filename)

        if os.stat(file_path).st_size == 0:
            print("Skipping empty file:", filename)
            continue

        print("Processing", filename)

        img = Image.open(file_path)
        temp = np.array(img)  # original for cropping
        grayscale = img.convert("L")

        _, thresh = cv2.threshold(np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV)

        rows, cols = thresh.shape
        indexStartX = indexEndX = 0
        indexStartY = indexEndY = 0

        # Vertical sweep
        flagx = 0
        for i in range(rows):
            line = thresh[i, :]
            if flagx == 0 and 255 in line:
                indexStartX = i
                flagx = 1
            elif flagx == 1 and 255 not in line:
                indexEndX = i
                break

        # Horizontal sweep
        flagy = 0
        for j in range(cols):
            line = thresh[indexStartX:indexEndX, j:j + 20]
            if flagy == 0 and 255 in line:
                indexStartY = j
                flagy = 1
            elif flagy == 1 and 255 not in line:
                indexEndY = j
                break

        # Safety check
        if indexEndX <= indexStartX or indexEndY <= indexStartY:
            print(f"  WARNING: No valid content found in {filename}, skipping.")
            continue

        # Draw rectangle (convert thresh to BGR for colored lines)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(
            thresh_bgr,
            (indexStartY, indexStartX),
            (indexEndY, indexEndX),
            (0, 0, 255),
            2,
        )

        # Crop
        temp_np = temp[indexStartX:indexEndX, indexStartY:indexEndY]

        # Save results
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LineSweep_Results")
        os.makedirs(path, exist_ok=True)

        out_crop = os.path.join(path, "LineSweep_Result_" + filename)
        cv2.imwrite(out_crop, temp_np)

        out_box = os.path.join(path, "LineSweep_Box_" + filename)
        cv2.imwrite(out_box, thresh_bgr)

        processed_files += 1

    print(f"{processed_files}/{total_files} files processed successfully")
    print("Processing Complete.")
    print("Check the LineSweep_Results folder for output.")


if __name__ == "__main__":
    main()
