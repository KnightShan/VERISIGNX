#!/usr/bin/env python3
"""
Robust run.py for signature verification evaluation.

Behaviors:
 - Prefers model_bundle.pkl (expected keys: model, scaler, voc, k).
 - Falls back to model.pkl (tries to detect if it's a wrapper or plain classifier).
 - Builds histograms using the vocabulary size available (bundle['k'] or per-group k).
 - Defensive handling of missing descriptors, small groups, SIFT availability, and empty preproc.
"""

import os
import pickle
from pathlib import Path
import numpy as np
from os import listdir
import cv2
from PIL import Image
import imagehash
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import preprocess
import features

# ---------- Config ----------
genuine_dir = "Data/genuine"
forged_dir = "Data/forged"
test_dir = "Data/origin"

num_groups = 29
DESIRED_K = 500  # fallback when no vocabulary found
MIN_IMAGES_FOR_SPLIT = 10  # original code assumed >=10 images per group

# model filenames
BUNDLE_PATH = Path("model_bundle.pkl")
MODEL_PATH = Path("model.pkl")
# ----------------------------

def list_images(folder):
    p = Path(folder)
    if not p.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in exts])

def safe_sift_create():
    try:
        return cv2.SIFT_create()
    except Exception:
        try:
            return cv2.xfeatures2d.SIFT_create()
        except Exception:
            return None

SIFT = safe_sift_create()
if SIFT is None:
    print("Warning: SIFT not available (opencv-contrib missing). Descriptor-based features will be empty.")

def sift_descriptors_from_image(img):
    """Return descriptors or None. img expected single-channel or 3-channel numpy array."""
    if SIFT is None:
        return None
    if img is None:
        return None
    # ensure uint8 and single-channel for SIFT
    if isinstance(img, np.ndarray):
        tmp = img
    else:
        # try to convert PIL image to array
        try:
            tmp = np.array(img)
        except Exception:
            return None
    if tmp.size == 0:
        return None
    if tmp.dtype != np.uint8:
        tmp = (np.clip(tmp, 0, 255)).astype(np.uint8)
    if tmp.ndim == 3:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    kp, des = SIFT.detectAndCompute(tmp, None)
    return des

# attempt to load model bundle or plain model
def load_model():
    """
    Returns tuple (model, scaler, voc, k)
    - model: classifier with .predict
    - scaler: StandardScaler or None
    - voc: numpy array of centroids or None
    - k: int vocabulary size or None
    """
    if BUNDLE_PATH.exists():
        try:
            b = pickle.load(open(BUNDLE_PATH, "rb"))
            model = b.get("model")
            scaler = b.get("scaler")
            voc = b.get("voc")
            k = b.get("k") if b.get("k") is not None else (voc.shape[0] if voc is not None else None)
            print(f"Loaded model_bundle.pkl (k={k})")
            return model, scaler, voc, k
        except Exception as e:
            print("Failed to load model_bundle.pkl:", e)

    if MODEL_PATH.exists():
        try:
            obj = pickle.load(open(MODEL_PATH, "rb"))
            # If it's a dict-like bundle
            if isinstance(obj, dict):
                model = obj.get("model") or obj.get("clf") or obj.get("classifier")
                scaler = obj.get("scaler")
                voc = obj.get("voc")
                k = obj.get("k") if obj.get("k") is not None else (voc.shape[0] if voc is not None else None)
                print("Loaded model.pkl as dict-like bundle.")
                return model, scaler, voc, k
            # If it's a class instance created by our train.py wrapper (ModelWrapper), try to extract attributes
            model = None; scaler = None; voc = None; k = None
            if hasattr(obj, "predict"):
                # Could be wrapper or plain classifier. Check for nested clf attribute.
                if hasattr(obj, "clf") and hasattr(obj.clf, "predict"):
                    model = obj.clf
                    scaler = getattr(obj, "scaler", None)
                    voc = getattr(obj, "voc", None)
                    k = getattr(obj, "k", None)
                    print("Loaded model.pkl: detected wrapper object; extracted inner classifier.")
                    return model, scaler, voc, k
                else:
                    # plain classifier
                    model = obj
                    print("Loaded model.pkl: found plain classifier.")
                    return model, None, None, None
            # fallback
            print("Loaded model.pkl but couldn't identify usable contents.")
            return None, None, None, None
        except Exception as e:
            # common error: can't get attribute ModelWrapper - handle politely
            print("Error loading model.pkl:", e)
            print("If error mentions missing class (e.g. ModelWrapper), run the unwrap helper or place the same wrapper class definition into this script.")
            return None, None, None, None

    print("No model_bundle.pkl or model.pkl found.")
    return None, None, None, None

# helper to extract ID from filename (same heuristic as your original)
def extract_id(name):
    try:
        return int(name.split('_')[0][-3:])
    except Exception:
        return None

# Build grouped lists
genuine_files = list_images(genuine_dir)
forged_files = list_images(forged_dir)
test_files = list_images(test_dir)

print("Total genuine:", len(genuine_files))
print("Total forged:", len(forged_files))
print("Total test:", len(test_files))

genuine_image_features = [[] for _ in range(num_groups)]
forged_image_features = [[] for _ in range(num_groups)]
image_features = []

for name in genuine_files:
    sid = extract_id(name)
    if sid and 1 <= sid <= num_groups:
        genuine_image_features[sid - 1].append({"name": name})
for name in forged_files:
    sid = extract_id(name)
    if sid and 1 <= sid <= num_groups:
        forged_image_features[sid - 1].append({"name": name})
for name in test_files:
    image_features.append({"name": name})

# load model once
model, model_scaler, model_voc, model_k = load_model()
if model is None:
    print("Fatal: Could not load any usable model. Exiting.")
    exit(1)

# main loop
cor = 0
wrong = 0

for i in range(num_groups):
    # collect training descriptors / features for this group
    des_list = []             # (path, des or None)
    im_contour_features = []  # list of 12-d vectors
    image_names = []

    # process genuine + forged for this group (keeps ordering)
    for bucket, folder in [(genuine_image_features[i], genuine_dir), (forged_image_features[i], forged_dir)]:
        for item in bucket:
            name = item["name"]
            image_path = os.path.join(folder, name)
            if not os.path.isfile(image_path):
                print(f"Missing file {image_path}, skipping")
                continue
            # preprocess
            try:
                pre_img = preprocess.preproc(image_path, display=False)
            except Exception as e:
                print(f"preprocess failed for {image_path}: {e}")
                pre_img = None

            # compute contour features (defensive)
            try:
                ar, b_area, ch_area, ct_area = features.get_contour_features(pre_img.copy() if pre_img is not None else pre_img, display=False)
            except Exception:
                ar = b_area = ch_area = ct_area = 0.0
            if b_area == 0:
                hull_over = 0.0
                contour_over = 0.0
            else:
                hull_over = ch_area / b_area
                contour_over = ct_area / b_area

            try:
                ratio = features.Ratio(pre_img.copy() if pre_img is not None else pre_img)
            except Exception:
                ratio = 0.0
            try:
                c0, c1 = features.Centroid(pre_img.copy() if pre_img is not None else pre_img)
            except Exception:
                c0 = c1 = 0.0
            try:
                ecc, sol = features.EccentricitySolidity(pre_img.copy() if pre_img is not None else pre_img)
            except Exception:
                ecc = sol = 0.0
            try:
                (skx, sky), (ktx, kty) = features.SkewKurtosis(pre_img.copy() if pre_img is not None else pre_img)
            except Exception:
                skx = sky = ktx = kty = 0.0

            im_contour_features.append([ar, hull_over, contour_over, ratio, c0, c1, ecc, sol, skx, sky, ktx, kty])

            # SIFT descriptors
            des = None
            try:
                des = sift_descriptors_from_image(pre_img)
            except Exception:
                des = None
            des_list.append((image_path, des))
            image_names.append(name)

    n_images = len(im_contour_features)
    if n_images == 0:
        print(f"Group {i+1}: no images, skipping")
        continue

    # only stack descriptors that exist
    all_descriptors = [d for _, d in des_list if d is not None and len(d) > 0]
    if len(all_descriptors) == 0:
        print(f"Group {i+1}: no descriptors at all (skipping vocabulary/training for this group)")
        continue
    try:
        descriptors = np.vstack(all_descriptors)
    except Exception as e:
        print("Error stacking descriptors for group", i+1, ":", e)
        continue

    # Decide vocabulary to use for this group's histograms:
    # Prefer model_voc/model_k (global vocabulary) if available; otherwise create per-group voc.
    if model_voc is not None and model_k is not None:
        voc = model_voc
        vocab_k = model_k
        # If voc has different descriptor dimension than descriptors, we must re-kmeans to match.
        if voc.shape[1] != descriptors.shape[1]:
            print(f"Group {i+1}: model voc descriptor dim {voc.shape[1]} != descriptors dim {descriptors.shape[1]}")
            # fall back to building per-group vocab
            voc = None
            vocab_k = None
    else:
        voc = None
        vocab_k = None

    if voc is None:
        effective_k = min(descriptors.shape[0], DESIRED_K)
        if effective_k < 1:
            print(f"Group {i+1}: not enough descriptors ({descriptors.shape[0]}), skipping")
            continue
        try:
            voc, _ = kmeans(descriptors, effective_k, 1)
            vocab_k = effective_k
            print(f"Group {i+1}: built per-group vocabulary k={vocab_k}")
        except Exception as e:
            print(f"Group {i+1}: kmeans failed: {e}")
            continue
    else:
        print(f"Group {i+1}: using model vocabulary k={vocab_k}")

    # build im_features of length vocab_k + 12
    im_features = np.zeros((n_images, vocab_k + 12), dtype="float32")
    for idx in range(n_images):
        des_i = des_list[idx][1]
        if des_i is not None and len(des_i) > 0:
            try:
                words, _ = vq(des_i, voc)
                for w in words:
                    if 0 <= int(w) < vocab_k:
                        im_features[idx][int(w)] += 1
            except Exception as e:
                # vq may fail if descriptor dims mismatch
                pass
        im_features[idx, vocab_k:vocab_k + 12] = im_contour_features[idx]

    # scale with scaler: prefer model_scaler if present, else fit local scaler
    if model_scaler is not None:
        try:
            im_features_scaled = model_scaler.transform(im_features)
            print(f"Group {i+1}: scaled with model scaler")
        except Exception as e:
            print(f"Group {i+1}: model scaler failed to transform (shape?) -> fitting local scaler. Error: {e}")
            stdSlr = StandardScaler().fit(im_features)
            im_features_scaled = stdSlr.transform(im_features)
    else:
        stdSlr = StandardScaler().fit(im_features)
        im_features_scaled = stdSlr.transform(im_features)

    # split as original (requires at least 10 images)
    if im_features_scaled.shape[0] < MIN_IMAGES_FOR_SPLIT:
        print(f"Group {i+1}: only {im_features_scaled.shape[0]} images (need >= {MIN_IMAGES_FOR_SPLIT}). Skipping.")
        continue

    train_genuine_features = im_features_scaled[0:3]
    test_genuine_features = im_features_scaled[3:5]
    train_forged_features = im_features_scaled[5:8]
    test_forged_features = im_features_scaled[8:10]

    # For prediction we expect model to accept vectors of length vocab_k + 12.
    expected_dim = vocab_k + 12
    if (test_genuine_features.shape[1] != expected_dim) or (test_forged_features.shape[1] != expected_dim):
        print(f"Group {i+1}: feature dimension mismatch (expected {expected_dim}, got {test_genuine_features.shape[1]}). Skipping.")
        continue

    # Model prediction
    try:
        genuine_res = model.predict(test_genuine_features)
        forged_res = model.predict(test_forged_features)
    except Exception as e:
        print(f"Group {i+1}: model.predict failed: {e}")
        continue

    # Count correctness (assumes labels: 2 -> genuine, 1 -> forged)
    for res in genuine_res:
        if int(res) == 2:
            cor += 1
        else:
            wrong += 1
    for res in forged_res:
        if int(res) == 1:
            cor += 1
        else:
            wrong += 1

# Report
if cor + wrong > 0:
    print("Final Accuracy SVM: {:.4f}".format(float(cor) / (cor + wrong)))
else:
    print("No predictions made; check data/model/splits.")
