try:
    from PIL import Image
    PILImage = Image
except ImportError:
    import Image
    PILImage = Image

import pytesseract
import cv2
import os
import numpy as np
from pathlib import Path

possible_tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(possible_tess_path):
    pytesseract.pytesseract.tesseract_cmd = possible_tess_path
else:
    pass

images_dir = "../Dataset/IDRBT_Cheque_Image_Dataset"

script_dir = Path(__file__).resolve().parent
if os.path.isabs(images_dir):
    input_path = Path(images_dir)
else:
    input_path = (script_dir / images_dir).resolve()

if not input_path.exists():
    print(f"Input folder does not exist: {input_path}")
    print("Create the folder and add images, or correct images_dir.")
    exit(1)

output_path = script_dir / "OCR_Results"
output_path.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

total_files = 0
processed_files = 0

for filename in sorted(os.listdir(input_path)):
    total_files += 1
    file_path = input_path / filename
    if file_path.suffix.lower() not in ALLOWED_EXT:
        continue

    print("OCR Processing file -", filename)
    img = cv2.imread(str(file_path))
    if img is None:
        print(f"  WARNING: cv2.imread failed for {filename}, skipping.")
        continue

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([103, 79, 60])
    upper = np.array([129, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 10:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    mask = 255 - mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    tdata = pytesseract.image_to_data(PILImage.open(str(file_path)))
    lines = tdata.splitlines()
    pleaseCd = [0, 0, 0, 0]
    aboveCd = [0, 0, 0, 0]
    please = 0
    above = 0
    sign = 0

    for d in lines[1:]:
        parts = d.split("\t")
        if len(parts) != 12:
            continue
        left, top, width, height, text = parts[6], parts[7], parts[8], parts[9], parts[11].strip()
        if not text:
            continue
        tl = text.lower()
        if tl == "please":
            try:
                pleaseCd = [int(left), int(top), int(width), int(height)]
                please += 1
            except ValueError:
                pass
        elif tl == "above":
            try:
                aboveCd = [int(left), int(top), int(width), int(height)]
                above += 1
            except ValueError:
                pass
        elif tl == "sign":
            sign += 1

        if len(text) == 11:
            prefix = text[:4]
            if prefix in {"SYNB", "SBIN", "HDFC", "CNRB", "PUNB", "UTIB", "ICIC"}:
                print("IFSC CODE : ", text)
            if prefix == "1C1C":
                l = list(text)
                l[0] = "I"
                l[2] = "I"
                print("IFSC CODE (fixed) : ", "".join(l))

    if please == 0 or above == 0:
        print(f"  INFO: 'please' or 'above' not detected for {filename}. Skipping crop.")
        continue

    lengthSign = (aboveCd[0] + aboveCd[3]) - pleaseCd[0]
    if lengthSign <= 0:
        print(f"  WARNING: Invalid computed lengthSign for {filename}. Skipping.")
        continue

    scaleY = 2
    scaleXL = 2.5
    scaleXR = 0.5

    x1 = int(pleaseCd[0] - lengthSign * scaleXL)
    y1 = int(pleaseCd[1] - lengthSign * scaleY)
    crop_w = int((scaleXL + scaleXR + 1) * lengthSign)
    crop_h = int(scaleY * lengthSign)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + max(1, crop_w))
    y2 = min(h, y1 + max(1, crop_h))

    if x2 <= x1 or y2 <= y1:
        print(f"  WARNING: invalid crop box for {filename}, skipping.")
        continue

    cropImg = img[y1:y2, x1:x2]
    if cropImg is None or cropImg.size == 0:
        print(f"  WARNING: empty crop for {filename}, skipping.")
        continue

    out_name = f"OCR_Result_{filename}"
    out_path = output_path / out_name
    ok = cv2.imwrite(str(out_path), cropImg)
    if ok:
        processed_files += 1
    else:
        print(f"  ERROR: failed to write {out_path}")

print(f"{processed_files}/{total_files} files processed successfully.")
print("Processing Complete. Output folder:", output_path)
