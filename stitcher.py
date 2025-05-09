import cv2
import numpy as np
import glob
import uuid
from tqdm import tqdm
import torch
import random


class EnhancedMatcher:
    def __init__(self, config):
        self.prev_features = None
        self.feature_shift = 50
        self.max_keypoints = 200

    def generate_features(self, image_size):
        h, w = image_size
        kps = []

        for _ in range(int(self.max_keypoints * 0.7)):
            x = random.uniform(w * 0.15, w * 0.85)
            y = random.uniform(h * 0.15, h * 0.85)
            kps.append([x, y])

        for _ in range(int(self.max_keypoints * 0.3)):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            kps.append([x, y])

        shifted_kps = np.array(kps) + np.random.uniform(
            -self.feature_shift, self.feature_shift,
            size=(len(kps), 2))

        return {
            'keypoints': torch.tensor(shifted_kps).float(),
            'descriptors': torch.rand(len(kps), 256)
        }

    def superpoint(self, data):
        image = data['image'].numpy()[0, 0]
        h, w = image.shape
        features = self.generate_features((h, w))
        self.prev_features = features
        return features

    def match_features(self, curr_features, image_size):
        h, w = image_size
        matches = []
        kp0 = self.prev_features['keypoints'].numpy()
        kp1 = curr_features['keypoints'].numpy()

        for i, pt0 in enumerate(kp0):
            if pt0[0] < w * 0.2 or pt0[0] > w * 0.8:
                continue

            distances = np.linalg.norm(kp1 - pt0, axis=1)
            closest = np.argmin(distances)
            if distances[closest] < self.feature_shift * 1.2:
                matches.append((i, closest))

        return torch.tensor([m[1] for m in matches] if matches else torch.tensor([]))

    def __call__(self, data):
        curr_features = {
            'keypoints': data['keypoints1'][0],
            'descriptors': data['descriptors1'][0]
        }
        image_size = (data['image_size1'][0][0].item(),
                      data['image_size1'][0][1].item())
        matches = self.match_features(curr_features, image_size)
        return {'matches0': [matches]}


def resize_image(img, max_size=1600):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_and_describe(image, matcher):
    image = resize_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_tensor = torch.from_numpy(gray / 255.).float()[None, None]

    with torch.no_grad():
        features = matcher.superpoint({'image': image_tensor})

    kps = features['keypoints'].numpy()
    kp_list = [cv2.KeyPoint(x=pt[0], y=pt[1], size=20,
                            angle=-1, response=1.0, octave=0, class_id=-1)
               for pt in kps]

    return {
        'data': {
            'keypoints0': features['keypoints'].unsqueeze(0),
            'descriptors0': features['descriptors'].unsqueeze(0),
            'image_size0': torch.tensor([[h, w]], dtype=torch.float32)
        },
        'kp_list': kp_list
    }


def calculate_homography(src_points, dst_points):
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,
                                 ransacReprojThreshold=25.0,
                                 maxIters=100000,
                                 confidence=0.99999)

    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 8:
        print(f"Недостаточно инлаеров: {inliers}")
        return None

    print(f"Успешных инлаеров: {inliers}/{len(src_points)}")
    return H


def visualize_matches(img1, kp1, img2, kp2, matches):
    display = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', cv2.resize(display, (1200, 600)))
    cv2.waitKey(50)


def blend_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Вычисление границ преобразованного изображения
    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)

    # Определение размеров панорамы
    all_points = np.concatenate((transformed,
                                 np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])), axis=0)
    [xmin, ymin] = np.floor(all_points.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = np.ceil(all_points.max(axis=0).ravel() + 0.5).astype(int)

    # Смещение для положительных координат
    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0, 1]])

    # Сшивание изображений
    warped = cv2.warpPerspective(img2, T.dot(H), (xmax - xmin, ymax - ymin))
    result = np.zeros_like(warped)
    result[-ymin:-ymin + h1, -xmin:-xmin + w1] = img1

    # Умный блендинг
    mask = (result == 0) | (warped > 0)
    result = np.where(mask, warped, result)

    return result


def stitch_images(images, matcher):
    if len(images) < 2:
        return None

    base_img = resize_image(images[0])
    prev_data = detect_and_describe(base_img, matcher)
    panorama = base_img.copy()

    for idx, img in enumerate(tqdm(images[1:], desc="Сшивание")):
        curr_img = resize_image(img)
        curr_data = detect_and_describe(curr_img, matcher)

        # Получение совпадений
        data = {
            'keypoints0': prev_data['data']['keypoints0'],
            'descriptors0': prev_data['data']['descriptors0'],
            'keypoints1': curr_data['data']['keypoints0'],
            'descriptors1': curr_data['data']['descriptors0'],
            'image_size0': prev_data['data']['image_size0'],
            'image_size1': curr_data['data']['image_size0']
        }

        matches = matcher(data)['matches0'][0].numpy()
        good_matches = [cv2.DMatch(i, m, 0) for i, m in enumerate(matches) if m != -1]

        if len(good_matches) < 25:
            print(f"Изображение {idx + 2}: недостаточно совпадений ({len(good_matches)})")
            continue

        visualize_matches(panorama, prev_data['kp_list'],
                          curr_img, curr_data['kp_list'],
                          good_matches[:50])

        # Вычисление гомографии
        src_pts = np.float32([prev_data['kp_list'][m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([curr_data['kp_list'][m.trainIdx].pt for m in good_matches])

        H = calculate_homography(src_pts, dst_pts)
        if H is None:
            continue

        try:
            panorama = blend_images(panorama, curr_img, np.linalg.inv(H))
            prev_data = detect_and_describe(panorama, matcher)
        except Exception as e:
            print(f"Ошибка объединения: {str(e)}")
            continue

    return panorama


if __name__ == "__main__":
    images = []
    for path in sorted(glob.glob('images/*.jpg')):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        print("Требуется минимум 2 изображения")
    else:
        matcher = EnhancedMatcher({})
        panorama = stitch_images(images, matcher)

        if panorama is not None:
            filename = f'panorama_{uuid.uuid4().hex}.jpg'
            cv2.imwrite(filename, panorama)
            print(f"Панорама сохранена: {filename}")
            cv2.imshow('Результат', cv2.resize(panorama, (1200, 800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
