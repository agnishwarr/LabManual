import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
imgA_color = cv2.imread("left.jpg")
imgB_color = cv2.imread("right.jpg")
grayA = cv2.cvtColor(imgA_color, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB_color, cv2.COLOR_BGR2GRAY)

# Custom SIFT-like feature detection
def detect_sift_keypoints_and_descriptors(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # ✅ Fixed sqrt warning

    keypoints = []
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            patch = magnitude[y - 1:y + 2, x - 1:x + 2]
            if magnitude[y, x] == np.max(patch) and magnitude[y, x] > 100:
                keypoints.append((x, y))

    descriptors = []
    for x, y in keypoints:
        if x - 8 < 0 or y - 8 < 0 or x + 8 >= img.shape[1] or y + 8 >= img.shape[0]:
            continue
        patch = img[y - 8:y + 8, x - 8:x + 8]
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx) * 180 / np.pi % 360
        hist = np.zeros(128)
        for i in range(4):
            for j in range(4):
                cell_mag = magnitude[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
                cell_ang = angle[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
                h, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                hist[i * 32 + j * 8:i * 32 + j * 8 + 8] = h
        norm = np.linalg.norm(hist)
        if norm != 0:
            hist = hist / norm
        descriptors.append(hist)

    return keypoints, np.array(descriptors)

# Detect features
kpA, descA = detect_sift_keypoints_and_descriptors(grayA)
kpB, descB = detect_sift_keypoints_and_descriptors(grayB)

print(f"Detected keypoints: Left = {len(kpA)}, Right = {len(kpB)}")

# Match descriptors
def match_descriptors(desc1, desc2, ratio=0.75):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = [np.linalg.norm(d1 - d2) for d2 in desc2]
        if len(distances) < 2:
            continue
        sorted_idx = np.argsort(distances)
        if distances[sorted_idx[0]] < ratio * distances[sorted_idx[1]]:
            matches.append((i, sorted_idx[0]))
    return matches

matches = match_descriptors(descA, descB)
print(f"Number of good matches: {len(matches)}")

# Exit if insufficient matches
if len(matches) < 4:
    print("Not enough matches to compute homography. Exiting.")
    exit()

# Extract point coordinates
src_pts = [kpA[m[0]] for m in matches]
dst_pts = [kpB[m[1]] for m in matches]

# Compute homography
def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[-1, -1]

def ransac_homography(src_pts, dst_pts, threshold=5.0, iterations=1000):
    best_H = None
    max_inliers = []
    for _ in range(iterations):
        idx = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = [src_pts[i] for i in idx]
        dst_sample = [dst_pts[i] for i in idx]
        H = compute_homography(src_sample, dst_sample)
        inliers = []
        for i in range(len(src_pts)):
            pt1 = np.array([*src_pts[i], 1.0])
            projected = H @ pt1
            projected /= projected[2]
            dist = np.linalg.norm(projected[:2] - np.array(dst_pts[i]))
            if dist < threshold:
                inliers.append(i)
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H
    return best_H, max_inliers

H, inliers = ransac_homography(src_pts, dst_pts)

# Image warping and stitching
def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_img2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32)
    corners_img2 = np.concatenate([corners_img2, np.ones((4, 1))], axis=1).T
    warped_corners = H @ corners_img2
    warped_corners /= warped_corners[2]
    warped_corners = warped_corners[:2].T
    all_corners = np.vstack((warped_corners, [[0, 0], [0, h1], [w1, h1], [w1, 0]]))
    [xmin, ymin] = np.floor(all_corners.min(axis=0)).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    result = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            p = np.array([x + xmin, y + ymin, 1])
            invH = np.linalg.inv(translation @ H)
            p_ = invH @ p
            p_ /= p_[2]
            xi, yi = int(round(p_[0])), int(round(p_[1]))
            if 0 <= xi < img2.shape[1] and 0 <= yi < img2.shape[0]:
                result[y, x] = img2[yi, xi]
    tx, ty = -xmin, -ymin
    result[ty:ty + h1, tx:tx + w1] = img1
    return result

stitched = warp_images(imgA_color, imgB_color, H)

# Visualization
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(imgA_color, cv2.COLOR_BGR2RGB))
plt.title("Left Image")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(imgB_color, cv2.COLOR_BGR2RGB))
plt.title("Right Image")
plt.axis('off')

key_imgA = imgA_color.copy()
key_imgB = imgB_color.copy()
for x, y in kpA:
    cv2.circle(key_imgA, (x, y), 1, (0, 255, 0), -1)
for x, y in kpB:
    cv2.circle(key_imgB, (x, y), 1, (0, 255, 0), -1)

plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(key_imgA, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints (Left)")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(key_imgB, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints (Right)")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.axis('off')

plt.tight_layout()
plt.show()
