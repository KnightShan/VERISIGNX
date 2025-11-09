import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
import cv2
import preprocess
import features

GenuineFolder = "Data/genuine"
ForgedFolder  = "Data/forged"

DESIRED_K = 500
MAX_DESCRIPTOR_SAMPLE = 20000
KMEANS_ITERS = 1

MODEL_PATH = Path("model.pkl")

def list_images(folder):
    p = Path(folder)
    if not p.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return [str(p / f) for f in sorted(os.listdir(p)) if Path(f).suffix.lower() in exts]

def safe_sift_descriptors(img):
    try:
        sift = cv2.SIFT_create()
    except Exception:
        try:
            sift = cv2.xfeatures2d.SIFT_create()
        except Exception:
            return None
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255)).astype(np.uint8)
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    _, des = sift.detectAndCompute(img_gray, None)
    return des

class ModelWrapper:
    def __init__(self, clf, scaler, voc, k):
        self.clf = clf
        self.scaler = scaler
        self.voc = voc
        self.k = k

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        expected_dim = (self.k + 12)
        if X.shape[1] != expected_dim:
            raise ValueError(f"Feature dimension mismatch: model expects {expected_dim} but got {X.shape[1]}")

        col_means = np.mean(X, axis=0)
        if np.max(np.abs(col_means)) < 1e-3:
            return self.clf.predict(X)
        else:
            Xs = self.scaler.transform(X)
            return self.clf.predict(Xs)

    def descriptors_to_histogram(self, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.k, dtype=float)
        words, _ = vq(descriptors, self.voc)
        hist = np.zeros(self.k, dtype=float)
        for w in words:
            if 0 <= int(w) < self.k:
                hist[int(w)] += 1
        return hist

def build_features_for_paths(paths, voc, k):
    feats = []
    for p in paths:
        try:
            pre = preprocess.preproc(p, display=False)
        except Exception:
            pre = None

        try:
            ar, b_area, ch_area, ct_area = features.get_contour_features(pre.copy(), display=False)
        except Exception:
            ar = b_area = ch_area = ct_area = 0.0
        if b_area == 0:
            hull_over_bound = 0.0
            contour_over_bound = 0.0
        else:
            hull_over_bound = ch_area / b_area
            contour_over_bound = ct_area / b_area
        try:
            ratio = features.Ratio(pre.copy())
        except Exception:
            ratio = 0.0
        try:
            c0, c1 = features.Centroid(pre.copy())
        except Exception:
            c0 = c1 = 0.0
        try:
            ecc, sol = features.EccentricitySolidity(pre.copy())
        except Exception:
            ecc = sol = 0.0
        try:
            (skx, sky), (ktx, kty) = features.SkewKurtosis(pre.copy())
        except Exception:
            skx = sky = ktx = kty = 0.0

        contour12 = [ar, hull_over_bound, contour_over_bound,
                     ratio, c0, c1, ecc, sol, skx, sky, ktx, kty]

        des = safe_sift_descriptors(pre)
        if des is None or len(des) == 0:
            hist = np.zeros(k, dtype=float)
        else:
            words, _ = vq(des, voc)
            hist = np.zeros(k, dtype=float)
            for w in words:
                if 0 <= int(w) < k:
                    hist[int(w)] += 1

        feat = np.concatenate([hist, np.array(contour12, dtype=float)])
        feats.append(feat)
    return np.vstack(feats)

def main():
    genuine = list_images(GenuineFolder)
    forged  = list_images(ForgedFolder)

    if len(genuine) == 0 and len(forged) == 0:
        print("No training images found in Data/genuine or Data/forged. Exiting.")
        return

    print(f"Found {len(genuine)} genuine and {len(forged)} forged images.")

    all_paths = genuine + forged
    labels = np.array([2]*len(genuine) + [1]*len(forged))

    per_image_des = []
    descriptor_pool = []
    print("Extracting SIFT descriptors and contour features (this may take time)...")
    for p in all_paths:
        try:
            pre = preprocess.preproc(p, display=False)
        except Exception:
            pre = None
        des = safe_sift_descriptors(pre)
        per_image_des.append(des)
        if des is not None and len(des) > 0:
            descriptor_pool.append(des)

    if len(descriptor_pool) == 0:
        raise SystemExit("No SIFT descriptors found in training set. Ensure opencv-contrib-python is installed and preprocess produces usable images.")

    descriptors_all = np.vstack(descriptor_pool)
    total_desc = descriptors_all.shape[0]
    print("Total SIFT descriptors collected:", total_desc)

    if total_desc > MAX_DESCRIPTOR_SAMPLE:
        idxs = np.random.choice(total_desc, MAX_DESCRIPTOR_SAMPLE, replace=False)
        descriptors_for_kmeans = descriptors_all[idxs]
    else:
        descriptors_for_kmeans = descriptors_all

    effective_k = min(DESIRED_K, descriptors_for_kmeans.shape[0])
    if effective_k < 2:
        raise SystemExit("Not enough descriptors to build vocabulary (k < 2).")

    print(f"Running kmeans to build vocabulary with k = {effective_k} ...")
    voc, var = kmeans(descriptors_for_kmeans.astype(float), effective_k, KMEANS_ITERS)
    print("kmeans complete. voc shape:", voc.shape)

    print("Building final feature matrix...")
    X = build_features_for_paths(all_paths, voc, effective_k)
    print("Feature matrix shape:", X.shape)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    print("Training LinearSVC...")
    clf = LinearSVC(max_iter=20000)
    clf.fit(Xs, labels)
    print("Training finished.")

    wrapper = ModelWrapper(clf, scaler, voc, effective_k)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(wrapper, f)
    print(f"Saved single model file: {MODEL_PATH} (contains classifier + scaler + vocab).")
    print("Model summary: k =", effective_k, "feature_dim =", effective_k + 12)
    print("Done.")

if __name__ == "__main__":
    main()
