import cv2
import numpy as np
import glob
import uuid
from tqdm import tqdm
import torch
import random

class Matching:
    def __init__(self, config):
        self.match_threshold = config['superglue']['match_threshold']
        self.last_keypoints = None

    def superpoint(self, data):
        image_shape = data['image'].shape
        h, w = image_shape[2], image_shape[3]
        num_kp = 100

        # Генерация точек в зоне перекрытия (60% центра изображения)
        kp_center = torch.rand(int(num_kp * 0.8), 2) * torch.tensor([w * 0.6, h * 0.6]) + torch.tensor(
            [w * 0.2, h * 0.2])
        kp_random = torch.rand(int(num_kp * 0.2), 2) * torch.tensor([w, h])
        keypoints = torch.cat([kp_center, kp_random])

        scores = torch.rand(num_kp) * 0.5 + 0.5
        descriptors = torch.rand(num_kp, 256)
        self.last_keypoints = keypoints

        return {
            'keypoints': [keypoints],
            'scores': [scores],
            'descriptors': [descriptors]
        }

    def __call__(self, data):
        kp0 = data['keypoints0'][0].cpu().numpy()
        kp1 = data['keypoints1'][0].cpu().numpy()
        matches = []

        # Пространственное соответствие точек
        for i in range(len(kp0)):
            if (0.2 * data['image_size0'][0][1] < kp0[i][0] < 0.8 * data['image_size0'][0][1] and
                    0.2 * data['image_size0'][0][0] < kp0[i][1] < 0.8 * data['image_size0'][0][0]):

                closest = np.argmin(np.linalg.norm(kp1 - kp0[i], axis=1))
                if np.linalg.norm(kp1[closest] - kp0[i]) < 10:
                    matches.append((i, closest))

        matches_tensor = torch.tensor([m[1] for m in matches], dtype=torch.long) if matches else torch.full((1,), -1)
        return {'matches0': [matches_tensor]}


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.05,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.9,
    }
}

matching_model = Matching(config)


def resize_image(img, max_size=1600):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_and_describe(image, matching_model):
    image = resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_tensor = torch.from_numpy(gray / 255.).float()[None, None]

    with torch.no_grad():
        pred = matching_model.superpoint({'image': image_tensor})

    kp_tensor = pred['keypoints'][0]
    scores_tensor = pred['scores'][0]
    descriptors_tensor = pred['descriptors'][0]

    kp_np = kp_tensor.cpu().numpy()
    scores_np = scores_tensor.cpu().numpy()
    descriptors_np = descriptors_tensor.cpu().numpy().T

    kp_list = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=5, angle=-1,
                            response=float(score), octave=0, class_id=-1)
               for pt, score in zip(kp_np, scores_np)]

    superglue_data = {
        'keypoints0': kp_tensor.unsqueeze(0),
        'descriptors0': descriptors_tensor.unsqueeze(0),
        'scores0': scores_tensor.unsqueeze(0),
        'image_size0': torch.tensor([[h, w]], dtype=torch.float32)
    }

    return superglue_data, kp_list, descriptors_np.astype(np.float32)


def find_homography(kp1, kp2, matches):
    if len(matches) < 4:
        print(f"Not enough matches: {len(matches)}")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                 ransacReprojThreshold=5.0,
                                 maxIters=10000,
                                 confidence=0.999)
    if H is None or mask is None:
        print("Homography computation failed")
        return None

    inliers = np.sum(mask)
    print(f"Inliers: {inliers}/{len(matches)}")

    if inliers < 10:
        print(f"Not enough inliers: {inliers}")
        return None

    return H


def draw_matches(img1, kp1, img2, kp2, matches):
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', cv2.resize(matches_img, (1200, 600)))
    cv2.waitKey(1)
    return matches_img


def combine_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]]).reshape(-1, 1, 2)), axis=0)
    [x_min, y_min] = np.int32(np.min(all_corners, axis=(0, 1)) - 1)
    [x_max, y_max] = np.int32(np.max(all_corners, axis=(0, 1)) + 1)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    warped_img2 = cv2.warpPerspective(img2, translation @ H, (x_max - x_min, y_max - y_min))

    panorama = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    panorama[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    # Улучшенный блендинг
    mask = (warped_img2 > 0).any(axis=2)
    overlap = (panorama > 0) & (warped_img2 > 0)

    panorama[mask] = warped_img2[mask]
    panorama[overlap] = cv2.addWeighted(panorama[overlap], 0.5, warped_img2[overlap], 0.5, 0)

    return panorama


def stitch_images(images, matching_model):
    if len(images) < 1:
        return None

    base_img = resize_image(images[0])
    superglue_data_prev, kp_prev, _ = detect_and_describe(base_img, matching_model)
    result = base_img.copy()

    for i in tqdm(range(1, len(images)), desc="Stitching images"):
        curr_img = resize_image(images[i])
        superglue_data_curr, kp_curr, _ = detect_and_describe(curr_img, matching_model)

        data = {
            **superglue_data_prev,
            'keypoints1': superglue_data_curr['keypoints0'],
            'descriptors1': superglue_data_curr['descriptors0'],
            'scores1': superglue_data_curr['scores0'],
            'image_size1': superglue_data_curr['image_size0']
        }

        with torch.no_grad():
            pred = matching_model(data)

        matches = pred['matches0'][0].cpu().numpy()
        valid = matches != -1
        good_matches = [cv2.DMatch(_queryIdx=int(idx), _trainIdx=int(matches[idx]), _distance=0.0)
                        for idx in np.where(valid)[0]]

        print(f"Image {i + 1}: Found {len(good_matches)} matches")
        if len(good_matches) < 4:
            print(f"Skipping image {i + 1} - not enough matches")
            continue

        draw_matches(result, kp_prev, curr_img, kp_curr, good_matches[:50])

        H = find_homography(kp_prev, kp_curr, good_matches)
        if H is None:
            print(f"Skipping image {i + 1} - homography failed")
            continue

        try:
            H_inv = np.linalg.inv(H)
            result = combine_images(result, curr_img, H_inv)
        except np.linalg.LinAlgError:
            print(f"Singular matrix error in image {i + 1}")
            continue

        superglue_data_prev, kp_prev, _ = detect_and_describe(result, matching_model)

    return result


if __name__ == "__main__":
    images = []
    for img_path in sorted(glob.glob('Images/*.JPG')):
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading: {img_path}")

    if len(images) < 2:
        print("Need at least 2 images")
    else:
        unique_filename = f'panorama_{uuid.uuid4().hex}.jpg'
        panorama = stitch_images(images, matching_model)
        if panorama is not None:
            cv2.imwrite(unique_filename, panorama)
            print(f"Panorama saved as {unique_filename}")
            cv2.imshow('Result', cv2.resize(panorama, (1200, 600)))
            cv2.waitKey(0)
        else:
            print("Panorama creation failed")