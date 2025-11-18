import cv2
import numpy as np
from sklearn.metrics import silhouette_score
import os

video_path = "white_coat_mice_30-min.mp4"
output_dir = "keyframes"
frame_interval = 10
hist_bins = 64
max_iters = 200
k_range = range(4,16)

print("starting...")
i=0
while True: 
    if not os.path.exists(output_dir+str(i)):
        os.makedirs(output_dir+str(i), exist_ok=True)
        break
    else:
        i+=1
output_dir = output_dir+str(i)

cap = cv2.VideoCapture(video_path)
frames, features = [], []
idx = 0

print("Got video1")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % frame_interval == 0:
        orig_frame = frame.copy()
        frame = frame[50:701, 240:1001]
        crop_offset_y, crop_offset_x = 50, 240

        tolerance = 5
        target_bgr = np.array([17, 29, 40])
        lower = np.array([max(target_bgr[0]-tolerance,0), max(target_bgr[1]-tolerance,0), max(target_bgr[2]-tolerance,0)])
        upper = np.array([min(target_bgr[0]+tolerance,255), min(target_bgr[1]+tolerance,255), min(target_bgr[2]+tolerance,255)])
        mask = cv2.inRange(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 130
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            #print(f"Contour area: {area}")
            if area >= min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # draw X on mask
                    size = 10
                    cv2.line(mask, (cX - size, cY - size), (cX + size, cY + size), (255), 2)
                    cv2.line(mask, (cX - size, cY + size), (cX + size, cY - size), (255), 2)

                    # translate to original frame coordinates
                    cX_full = cX + crop_offset_x
                    cY_full = cY + crop_offset_y

                    half_size = 300
                    x1 = max(cX_full - half_size, 0)
                    y1 = max(cY_full - half_size, 0)
                    x2 = min(cX_full + half_size, orig_frame.shape[1]-1)
                    y2 = min(cY_full + half_size, orig_frame.shape[0]-1)
                    cropped = orig_frame[y1:y2+1, x1:x2+1]
                    h, w = cropped.shape[:2]
                    size_pad = max(h, w)
                    pad_vertical = size_pad - h
                    pad_horizontal = size_pad - w
                    top = pad_vertical // 2
                    bottom = pad_vertical - top
                    left = pad_horizontal // 2
                    right = pad_horizontal - left
                    cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                    cropped = cv2.resize(cropped, (320, 320))
                    small = cv2.resize(cropped, (160, 160))
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                        [hist_bins, hist_bins, hist_bins],
                                        [0, 180, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    frames.append(cropped)
                    features.append(hist)

        cv2.imshow("mask", mask)
        cv2.waitKey(1)
    idx += 1

cap.release()
features = np.array(features)

print("finished")

def kmeans_plus_plus_init(X, k):
    np.random.seed(42)
    n = X.shape[0]
    centers = [X[np.random.randint(0, n)]]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(x - c)**2 for c in centers) for x in X])
        probs = dists / dists.sum()
        next_idx = np.random.choice(n, p=probs)
        centers.append(X[next_idx])
    return np.array(centers)

best_k = None
best_score = -1
best_labels = None
best_centroids = None

# clustering
for k in k_range:
    print("starting k =", k)
    centroids = kmeans_plus_plus_init(features, k)
    for _ in range(max_iters):
        distances = np.linalg.norm(features[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            features[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    if len(set(labels)) > 1:
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centroids = centroids

print(f"optimal k = {best_k}")

representatives = []
for i in range(best_k):
    cluster_idx = np.where(best_labels == i)[0]
    if len(cluster_idx) == 0:
        continue
    cluster_feats = features[cluster_idx]
    distances = np.linalg.norm(cluster_feats - best_centroids[i], axis=1)
    best_idx = cluster_idx[np.argmin(distances)]
    representatives.append(best_idx)

for i, idx in enumerate(representatives):
    path = os.path.join(output_dir, f"cluster_{i+1}.jpg")
    cv2.imwrite(path, frames[idx])
    print(f"saved {path}")
