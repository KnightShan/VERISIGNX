from PIL import Image
import operator
import os
import random
import numpy as np
import cv2
from pathlib import Path
from UnionArray import *


def run(img):
    data = img.load()
    width, height = img.size

    uf = UFarray()
    labels = {}

    for y in range(height):
        for x in range(width):
            if data[x, y] == 255:
                continue

            if y > 0 and data[x, y - 1] == 0:
                labels[(x, y)] = labels[(x, y - 1)]
            elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:
                c = labels[(x + 1, y - 1)]
                labels[(x, y)] = c
                if x > 0 and data[x - 1, y - 1] == 0:
                    uf.union(c, labels[(x - 1, y - 1)])
                elif x > 0 and data[x - 1, y] == 0:
                    uf.union(c, labels[(x - 1, y)])
            elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
                labels[(x, y)] = labels[(x - 1, y - 1)]
            elif x > 0 and data[x - 1, y] == 0:
                labels[(x, y)] = labels[(x - 1, y)]
            else:
                labels[(x, y)] = uf.makeLabel()

    uf.flatten()

    colors = {}
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:
        component = uf.find(labels[(x, y)])
        labels[(x, y)] = component
        if component not in colors:
            colors[component] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        outdata[x, y] = colors[component]

    return labels, output_img


def cropByConnectedComponent(points, temp, filename):
    sig = {}
    for data in points.values():
        pts = np.array(data, dtype=np.int32)
        if pts.size == 0:
            continue
        x, y, w, h = cv2.boundingRect(pts)
        sig[(x, y, w, h)] = w * h

    if not sig:
        print(f"  WARNING: no valid components found for {filename}")
        return

    (x, y, w, h), _ = max(sig.items(), key=operator.itemgetter(1))
    temp_np = temp[y:y + h, x:x + w]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConnectedComponents_Results")
    os.makedirs(path, exist_ok=True)

    s1 = "CC_Result_" + filename
    cv2.imwrite(os.path.join(path, s1), temp_np)


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
        temp = np.array(img)

        grayscale = img.convert("L")
        _, thresh = cv2.threshold(np.array(grayscale), 127, 255, cv2.THRESH_BINARY_INV)

        img_bin = img.point(lambda p: 255 if p > 128 else 0).convert("1")

        labels, _ = run(img_bin)

        points = {}
        for k, v in labels.items():
            points.setdefault(v, []).append(k)

        cropByConnectedComponent(points, temp, filename)
        processed_files += 1

    print(f"{processed_files}/{total_files} files processed successfully")
    print("Processing Complete.")
    print("You may check the ConnectedComponents_Results folder in the same directory.")


if __name__ == "__main__":
    main()
