# app.py
import os
import io
import pickle
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import cv2
from PIL import Image
import pytesseract

# import your preprocessing/feature helpers
import preprocess   # uses your preprocess.preproc() from preprocess.py
import features     # uses functions from features.py

# --- Config ---
UPLOAD_FOLDER = "uploads"
OCR_CROP_FOLDER = "ocr_crops"
LINESWEEP_FOLDER = "linesweep"
MODEL_PATH = Path("model.pkl")   # expected to exist (or model_bundle handling could be added)
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}
SIFT = None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OCR_CROP_FOLDER, exist_ok=True)
os.makedirs(LINESWEEP_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Helper: safe SIFT creator (same idea as run.py safe creation) ---
def safe_sift_create():
    global SIFT
    if SIFT is not None:
        return SIFT
    try:
        SIFT = cv2.SIFT_create()
    except Exception:
        try:
            SIFT = cv2.xfeatures2d.SIFT_create()
        except Exception:
            SIFT = None
    return SIFT

# --- Provide ModelWrapper class declaration so pickle can unpickle wrapper objects created by train.py ---
class ModelWrapper:
    def __init__(self, clf=None, scaler=None, voc=None, k=None):
        self.clf = clf
        self.scaler = scaler
        self.voc = voc
        self.k = k

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        expected_dim = (self.k + 12) if (self.k is not None) else X.shape[1]
        # if scaler present and features are not already zero-mean, scale them
        if self.scaler is not None:
            col_means = np.mean(X, axis=0)
            if np.max(np.abs(col_means)) >= 1e-3:
                Xs = self.scaler.transform(X)
            else:
                Xs = X
        else:
            Xs = X
        return self.clf.predict(Xs)

    # helper to build histogram from descriptors (if wrapper stored voc)
    def descriptors_to_histogram(self, descriptors):
        from scipy.cluster.vq import vq
        if self.voc is None or self.k is None:
            return None
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.k, dtype=float)
        words, _ = vq(descriptors, self.voc)
        hist = np.zeros(self.k, dtype=float)
        for w in words:
            if 0 <= int(w) < self.k:
                hist[int(w)] += 1
        return hist

# --- Load the model (attempts to handle train.py wrapper or plain classifier) ---
def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Place your trained model.pkl in the project root.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # if obj is dict-like with keys
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("classifier")
        scaler = obj.get("scaler")
        voc = obj.get("voc")
        k = obj.get("k") if obj.get("k") is not None else (voc.shape[0] if voc is not None else None)
        # wrap into our ModelWrapper-like object
        wrapper = ModelWrapper(clf=model, scaler=scaler, voc=voc, k=k)
        return wrapper
    # If it's a wrapper instance (pickled from train.py), it will unpickle into our ModelWrapper class above
    if hasattr(obj, "predict") and hasattr(obj, "descriptors_to_histogram"):
        # already complete wrapper
        return obj
    # If it's the wrapper that stores clf/scaler/voc fields:
    if hasattr(obj, "clf") and hasattr(obj, "scaler"):
        # use our ModelWrapper to interact
        wrapper = ModelWrapper(clf=getattr(obj, "clf", None),
                               scaler=getattr(obj, "scaler", None),
                               voc=getattr(obj, "voc", None),
                               k=getattr(obj, "k", None))
        return wrapper
    # If it's a plain sklearn classifier:
    if hasattr(obj, "predict"):
        return ModelWrapper(clf=obj, scaler=None, voc=None, k=None)
    raise RuntimeError("Unrecognized model.pkl structure.")

MODEL = None
try:
    MODEL = load_model(MODEL_PATH)
    print("Model loaded (app).")
except Exception as e:
    print("Model load warning:", e)
    MODEL = None

# --- OCR-like crop function (simplified single-image adaptation of your OCR.py logic) ---
def ocr_crop_single(image_path, out_folder=OCR_CROP_FOLDER):
    """
    Attempts to find the 'Please ... above ... sign' region as in your OCR.py
    If it cannot, returns None. Otherwise returns path to cropped image.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]

    # create HSV mask similar to your script (these thresholds were in OCR.py)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([103, 79, 60])
    upper = np.array([129, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # morphological clean-up (remove small contours)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 10:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    mask_inv = 255 - mask
    mask_inv = cv2.GaussianBlur(mask_inv, (3, 3), 0)

    # OCR text parsing to find coordinates (use pytesseract)
    try:
        pil = Image.open(image_path)
        tdata = pytesseract.image_to_data(pil)
        lines = tdata.splitlines()
    except Exception:
        lines = []

    pleaseCd = None
    aboveCd = None
    sign_count = 0

    for d in lines[1:]:
        parts = d.split("\t")
        if len(parts) != 12:
            continue
        left, top, width, height, text = parts[6], parts[7], parts[8], parts[9], parts[11].strip()
        if not text:
            continue
        tl = text.lower()
        try:
            left_i = int(left); top_i = int(top); width_i = int(width); height_i = int(height)
        except Exception:
            continue
        if tl == "please":
            pleaseCd = (left_i, top_i, width_i, height_i)
        elif tl == "above":
            aboveCd = (left_i, top_i, width_i, height_i)
        elif tl == "sign":
            sign_count += 1

    # If OCR failed to find please/above, fallback to whole image
    if pleaseCd is None or aboveCd is None:
        # fallback: return whole image as crop
        out = os.path.join(out_folder, "ocr_crop_full_" + os.path.basename(image_path))
        cv2.imwrite(out, img_bgr)
        return out

    # compute crop around the detected "sign" area (replicates the math in OCR.py)
    lengthSign = (aboveCd[0] + aboveCd[2]) - pleaseCd[0]
    if lengthSign <= 0:
        # fallback to full image
        out = os.path.join(out_folder, "ocr_crop_full_" + os.path.basename(image_path))
        cv2.imwrite(out, img_bgr)
        return out

    scaleY = 2.0
    scaleXL = 2.5
    scaleXR = 0.5

    x1 = int(pleaseCd[0] - lengthSign * scaleXL)
    y1 = int(pleaseCd[1] - lengthSign * scaleY)
    crop_w = int((scaleXL + scaleXR + 1) * lengthSign)
    crop_h = int(scaleY * lengthSign)

    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + max(1, crop_w))
    y2 = min(h, y1 + max(1, crop_h))
    if x2 <= x1 or y2 <= y1:
        out = os.path.join(out_folder, "ocr_crop_full_" + os.path.basename(image_path))
        cv2.imwrite(out, img_bgr)
        return out

    cropImg = img_bgr[y1:y2, x1:x2]
    out = os.path.join(out_folder, "ocr_crop_" + os.path.basename(image_path))
    cv2.imwrite(out, cropImg)
    return out

# --- LineSweep single-image (adapted from your LineSweep.py) ---
def linesweep_crop_single(image_path, out_folder=LINESWEEP_FOLDER):
    img_pil = Image.open(image_path)
    temp = np.array(img_pil)
    grayscale = img_pil.convert("L")
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
        # use a small vertical slice like your script
        line = thresh[indexStartX:indexEndX, j:j + 20] if indexEndX > indexStartX else thresh[:, j:j+20]
        if flagy == 0 and 255 in line:
            indexStartY = j
            flagy = 1
        elif flagy == 1 and 255 not in line:
            indexEndY = j
            break

    if indexEndX <= indexStartX or indexEndY <= indexStartY:
        # fallback: return full image
        out = os.path.join(out_folder, "ls_full_" + os.path.basename(image_path))
        cv2.imwrite(out, temp)
        return out

    temp_np = temp[indexStartX:indexEndX, indexStartY:indexEndY]
    out_crop = os.path.join(out_folder, "ls_crop_" + os.path.basename(image_path))
    cv2.imwrite(out_crop, temp_np)
    return out_crop

# --- Build feature vector for a single image path (SIFT histogram + 12 contour features) ---
def build_feature_vector(image_path, model_wrapper):
    # 1) preprocess -> binary crop similarly to preprocess.preproc
    try:
        pre = preprocess.preproc(path=str(image_path), display=False)
    except Exception:
        pre = None

    # 2) contour 12 features (safe wrapper)
    try:
        ar, b_area, ch_area, ct_area = features.get_contour_features(pre.copy() if pre is not None else pre, display=False)
    except Exception:
        ar = b_area = ch_area = ct_area = 0.0
    if b_area == 0:
        hull_over = 0.0
        contour_over = 0.0
    else:
        hull_over = ch_area / b_area
        contour_over = ct_area / b_area
    try:
        ratio = features.Ratio(pre.copy() if pre is not None else pre)
    except Exception:
        ratio = 0.0
    try:
        c0, c1 = features.Centroid(pre.copy() if pre is not None else pre)
    except Exception:
        c0 = c1 = 0.0
    try:
        ecc, sol = features.EccentricitySolidity(pre.copy() if pre is not None else pre)
    except Exception:
        ecc = sol = 0.0
    try:
        (skx, sky), (ktx, kty) = features.SkewKurtosis(pre.copy() if pre is not None else pre)
    except Exception:
        skx = sky = ktx = kty = 0.0

    contour12 = [ar, hull_over, contour_over, ratio, c0, c1, ecc, sol, skx, sky, ktx, kty]

    # 3) SIFT descriptors (use safe generator)
    sift = safe_sift_create()
    des = None
    if sift is not None and pre is not None:
        try:
            # SIFT expects uint8 single channel
            tmp = pre.copy()
            if tmp.dtype != np.uint8:
                tmp = (np.clip(tmp, 0, 255)).astype(np.uint8)
            if tmp.ndim == 3:
                tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
            kp, des = sift.detectAndCompute(tmp, None)
        except Exception:
            des = None

    # 4) Build histogram using wrapper vocabulary if available
    hist = None
    if hasattr(model_wrapper, "descriptors_to_histogram"):
        hist = model_wrapper.descriptors_to_histogram(des)
    # If wrapper has no voc or hist is None, fallback to zeros (model might expect local k)
    if hist is None:
        k = getattr(model_wrapper, "k", None)
        if k is None:
            # unknown k: fallback to 500 (train.py DESIRED_K) -> will likely mismatch if model expects different dim
            k = 500
        hist = np.zeros(k, dtype=float)

    feat = np.concatenate([hist, np.array(contour12, dtype=float)])
    return feat

# --- Flask routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "photo" not in request.files:
        return "No file part", 400
    file = request.files["photo"]
    if file.filename == "":
        return "No selected file", 400
    fn = secure_filename(file.filename)
    ext = fn.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return "File type not allowed", 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
    file.save(save_path)

    # 1) OCR crop
    ocr_crop = ocr_crop_single(save_path)
    # 2) LineSweep crop of the OCR crop (if OCR gave something)
    ls_input = ocr_crop if ocr_crop is not None else save_path
    ls_crop = linesweep_crop_single(ls_input)

    # 3) Build feature vector from the linesweep crop
    if MODEL is None:
        return "Model not loaded on server. Place model.pkl in project root and restart.", 500
    feature_vector = build_feature_vector(ls_crop, MODEL)

    # 4) Predict
    try:
        pred = MODEL.predict(feature_vector)
        label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
    except Exception as e:
        return f"Model prediction failed: {e}", 500

    # Map labels (train.py used 2=genuine,1=forged)
    if label == 2:
        verdict = "Genuine"
    elif label == 1:
        verdict = "Fake / Forged"
    else:
        verdict = f"Unknown label ({label})"

    # Return a simple HTML page showing the uploaded, OCR crop and LineSweep crop and the verdict
    return render_template("index.html",
                           uploaded_image=url_for("uploaded_file", filename=fn),
                           ocr_crop=(os.path.basename(ocr_crop) if ocr_crop else None),
                           ls_crop=(os.path.basename(ls_crop) if ls_crop else None),
                           verdict=verdict)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
