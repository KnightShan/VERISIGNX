import numpy as np
import os
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
import imagehash
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
import preprocess as preproc
import features

genuine_dir = "Data/genuine"
forged_dir = "Data/forged"
image_test_dir = "Line Sweep\LineSweep_Results"

genuine_image_filenames = sorted(listdir(genuine_dir))
forged_image_filenames = sorted(listdir(forged_dir))
image_test_filenames = sorted(listdir(image_test_dir))

print("Total Number of Files in genuine folder:", len(genuine_image_filenames))
print("Total Number of Files in forged folder:", len(forged_image_filenames))
print("Total Number of Files in test folder:", len(image_test_filenames))

num_groups = 29
genuine_image_features = [[] for _ in range(num_groups)]
forged_image_features = [[] for _ in range(num_groups)]
image_features = []

def extract_id(name):
    try:
        return int(name.split('_')[0][-3:])
    except Exception:
        return None

for name in genuine_image_filenames:
    sid = extract_id(name)
    if sid and 1 <= sid <= num_groups:
        genuine_image_features[sid - 1].append({"name": name})

for name in forged_image_filenames:
    sid = extract_id(name)
    if sid and 1 <= sid <= num_groups:
        forged_image_features[sid - 1].append({"name": name})

for name in image_test_filenames:
    image_features.append({"name": name})

def preprocess_image(path):
    return preproc.preproc(path, display=False)

def sift_extract(im):
    try:
        sift_detector = cv2.SIFT_create()
    except Exception:
        try:
            sift_detector = cv2.xfeatures2d.SIFT_create()
        except Exception:
            return None
    kp, des = sift_detector.detectAndCompute(im, None)
    return des

cor = 0
wrong = 0

desired_k = 500

for group_idx in range(num_groups):
    des_list = []
    im_contour_features = []
    image_order_names = []

    for bucket, folder in [(genuine_image_features[group_idx], genuine_dir),
                           (forged_image_features[group_idx], forged_dir)]:
        for item in bucket:
            name = item["name"]
            path = os.path.join(folder, name)
            pre_img = preprocess_image(path)
            try:
                ph = imagehash.phash(Image.open(path))
                phash_int = int(str(ph), 16)
            except Exception:
                phash_int = 0
            ar, b_area, hull_area, contour_area = features.get_contour_features(pre_img.copy(), display=False)
            if b_area == 0:
                hull_over = 0.0
                contour_over = 0.0
            else:
                hull_over = hull_area / b_area
                contour_over = contour_area / b_area
            ratio = features.Ratio(pre_img.copy())
            c0, c1 = features.Centroid(pre_img.copy())
            ecc, sol = features.EccentricitySolidity(pre_img.copy())
            (skx, sky), (ktx, kty) = features.SkewKurtosis(pre_img.copy())

            im_contour_features.append([
                ar, hull_over, contour_over,
                ratio, c0, c1, ecc, sol, skx, sky, ktx, kty
            ])

            sift_input = pre_img.copy()
            if sift_input.dtype != np.uint8:
                sift_input = sift_input.astype(np.uint8)
            des = None
            try:
                des = sift_extract(sift_input)
            except Exception:
                des = None
            des_list.append(des)
            image_order_names.append(name)

    im_contour_features_test = []
    des_list_test = []
    test_names = []
    for item in image_features:
        name = item["name"]
        path = os.path.join(image_test_dir, name)
        pre_img = preprocess_image(path)
        ar, b_area, hull_area, contour_area = features.get_contour_features(pre_img.copy(), display=False)
        if b_area == 0:
            hull_over = 0.0
            contour_over = 0.0
        else:
            hull_over = hull_area / b_area
            contour_over = contour_area / b_area
        ratio = features.Ratio(pre_img.copy())
        c0, c1 = features.Centroid(pre_img.copy())
        ecc, sol = features.EccentricitySolidity(pre_img.copy())
        (skx, sky), (ktx, kty) = features.SkewKurtosis(pre_img.copy())

        im_contour_features_test.append([
            ar, hull_over, contour_over,
            ratio, c0, c1, ecc, sol, skx, sky, ktx, kty
        ])

        if pre_img.dtype != np.uint8:
            pre_img = pre_img.astype(np.uint8)
        des = None
        try:
            des = sift_extract(pre_img)
        except Exception:
            des = None
        des_list_test.append(des)
        test_names.append(name)

    all_des = [d for d in des_list if d is not None and len(d) > 0]
    if len(all_des) == 0:
        print(f"Group {group_idx+1}: no descriptors found, skipping group.")
        continue
    try:
        descriptors = np.vstack(all_des)
    except Exception as e:
        print(f"Group {group_idx+1}: error stacking descriptors: {e}")
        continue

    effective_k = min(descriptors.shape[0], desired_k)
    if effective_k < 1:
        print(f"Group {group_idx+1}: not enough descriptors ({descriptors.shape[0]}), skipping")
        continue

    voc, variance = kmeans(descriptors, effective_k, 1)

    n_train_images = len(im_contour_features)
    im_features = np.zeros((n_train_images, desired_k + 12), dtype="float32")
    for idx in range(n_train_images):
        des_i = des_list[idx]
        if des_i is not None and len(des_i) > 0:
            words, distance = vq(des_i, voc)
            for w in words:
                if 0 <= w < desired_k:
                    im_features[idx][w] += 1
        im_features[idx, desired_k:desired_k + 12] = im_contour_features[idx]

    n_test_images = len(im_contour_features_test)
    im_features_test = np.zeros((n_test_images, desired_k + 12), dtype="float32")
    for idx in range(n_test_images):
        des_i = des_list_test[idx]
        if des_i is not None and len(des_i) > 0:
            words, distance = vq(des_i, voc)
            for w in words:
                if 0 <= w < desired_k:
                    im_features_test[idx][w] += 1
        im_features_test[idx, desired_k:desired_k + 12] = im_contour_features_test[idx]

    scaler = StandardScaler().fit(im_features)
    im_features_scaled = scaler.transform(im_features)
    im_features_test_scaled = scaler.transform(im_features_test)

    if im_features_scaled.shape[0] < 10:
        print(f"Group {group_idx+1}: only {im_features_scaled.shape[0]} images (need >=10 for fixed split). Skipping group.")
        continue

    train_genuine_features = im_features_scaled[0:3]
    test_genuine_features = im_features_scaled[3:5]
    train_forged_features = im_features_scaled[5:8]
    test_forged_features = im_features_scaled[8:10]

    clf = LinearSVC(max_iter=5000)
    X_train = np.vstack((train_forged_features, train_genuine_features))
    y_train = np.array([1] * len(train_forged_features) + [2] * len(train_genuine_features))
    clf.fit(X_train, y_train)

    genuine_res = clf.predict(test_genuine_features)
    forged_res = clf.predict(test_forged_features)
    test_res = clf.predict(im_features_test_scaled)

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

if cor + wrong > 0:
    print("Final Accuracy SVM: {:.4f}".format(float(cor) / (cor + wrong)))
else:
    print("No predictions were made.")