import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import json
import pandas as pd
import shutil
import os
from skimage import transform as trans
import argparse

def alignment(src_img, src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (112, 112)
    src_pts = np.array(src_pts).reshape(3,2)

    s = np.array(src_pts).astype(np.float32)[:3]
    r = np.array(ref_pts).astype(np.float32)[:3]

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def parse_lst_line(line):
    vec = line.strip().split("\t")
    assert len(vec) >= 3
    aligned = int(vec[0])
    image_path = vec[1]
    label = int(vec[2])
    bbox = None
    landmark = None
    # print(vec)
    if len(vec) > 3:
        bbox = np.zeros((4,), dtype=np.int32)
        for i in range(3, 7):
            bbox[i - 3] = int(vec[i])
        landmark = None
        if len(vec) > 7:
            _l = []
            for i in range(7, 17):
                _l.append(float(vec[i]))
            landmark = np.array(_l).reshape((2, 5)).T
    # print(aligned)
    return image_path, label, bbox, landmark, aligned


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode == 'rgb':
            # print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def preprocess_malte(img, bbox, landmark, image_size):
    bbox = bbox[0]
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    rotate = 0
    _scale = 112 / (max(w, h) * 1.5)
    # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
    aimg, M = transform(img, center, 112, _scale, rotate)
    return aimg


def preprocess(img, bbox=None, landmark=None, **kwargs):
    if isinstance(img, str):
        img = read_image(img, **kwargs)
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        #src = np.array([
        #    [30.2946, 51.6963],
        #    [65.5318, 51.5014],
        #    [48.0252, 71.7366],
        #    [33.5493, 92.3655],
        #    [62.7299, 92.2041]], dtype=np.float32)[:3]
        src = np.array([
            [32.4682, 43.6963],
            [79.5318, 43.5014],
            [55.0252, 71.7366]], dtype=np.float32)
        #src = np.array([
        #    [30.2946, 51.6963],
        #    [65.5318, 51.5014],
        #    [48.0252, 71.7366]], dtype=np.float32)
        #if image_size[1] == 112:
        #    src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 20)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped

def convert_to_list(str_like_list):
    res = [list(map(float, i[1:-1].split(', '))) for i in str_like_list.split('\n')]
    return np.array(res)

def adjust_src_pts(src_pts):
    if src_pts[2][1] > src_pts[1][1] and src_pts[2][1] > src_pts[0][1]:
        return src_pts
    # up-side down
    if src_pts[2][1] < src_pts[1][1] and src_pts[2][1] < src_pts[0][1]:
        if src_pts[0][0] < src_pts[1][0]:
            src_pts[[0, 1]] = src_pts[[1, 0]]
    # right
    if src_pts[2][0] > src_pts[1][0] and src_pts[2][0] > src_pts[0][0]:
        if src_pts[0][1] < src_pts[1][1]:
            src_pts[[0, 1]] = src_pts[[1, 0]]
    # left
    if src_pts[2][0] < src_pts[1][0] and src_pts[2][0] < src_pts[0][0]:
        if src_pts[0][1] > src_pts[1][1]:
            src_pts[[0, 1]] = src_pts[[1, 0]]
    return src_pts

def process_face_alignment(opt):
    kpt_json_path, bbox_json_path = opt.kpt_json, opt.bbox_json
    
    if os.path.exists(opt.name):
        num = len([i for i in os.listdir('.') if opt.name in i])
        opt.name = f"{opt.name}_{num}"

    with open(kpt_json_path) as f:
        kpts_dict = json.load(f)
    with open(bbox_json_path) as f:
        bbox_dict = json.load(f)
    for source in list(kpts_dict.keys()):
        img = cv2.imread(source)

        img_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
        src_pts_list = kpts_dict[source]['kpts']
        bbox_list = kpts_dict[source]['bbox']
        bbox_list2 = bbox_dict[source]['bbox']
        
        save_path = f"{opt.name}/{kpts_dict[source]['folder']}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if src_pts_list and bbox_list:
            for i, (bbox, s) in enumerate(zip(bbox_list, src_pts_list)):
                src_pts = adjust_src_pts(np.array(s).reshape(3, 2))
                face_img = preprocess(img, bbox, src_pts, image_size="112, 112", margin=0)
                cv2.imwrite(f"{save_path}/{i}_alignment_{img_name}", face_img)
                print(f"Done: {save_path}/{i}_alignment_{img_name}")
        elif bbox_list2:
            for i, bbox in enumerate(bbox_list2):
                crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imwrite(f"{save_path}/{i}_crop_{img_name}", crop_img)
                print(f"Done: {save_path}/{i}_crop_{img_name}")
        else:
            print(f"Didn't find bboxes or kpts on {img_name}")
    
    print(f"All images save to: {opt.name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kpt-json', default='', help='kpt json file')
    parser.add_argument('--bbox-json', default='', help='bbox json file')
    parser.add_argument('--name', default='face_alignment_images', help='save results to project/name')
    opt = parser.parse_args()
    print(opt)
    process_face_alignment(opt)
    