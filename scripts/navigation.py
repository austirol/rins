import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

image = cv2.imread(os.path.join(os.getcwd(), "..", "maps", "mapademo3.pgm"), cv2.IMREAD_GRAYSCALE)
image_og = cv2.imread(os.path.join(os.getcwd(), "..", "maps", "mapademo3.pgm"), cv2.IMREAD_COLOR)
# print(np.unique(image))

# where = []
# for i in np.unique(image):
#     where.append(image == i)

# fig = plt.figure(figsize=(20, 5))
# for i, w in enumerate(where):
#     fig.add_subplot(1, len(where)+1, i+1)
#     plt.title(str(np.unique(image)[i]))
#     plt.imshow(w, cmap='gray')

# fig.add_subplot(1, len(where)+1, len(where)+1)
# plt.imshow(image)
# plt.show()

# točka ne sme bit na 0 in 205, lahko je samo na 254

# valid_space = np.pad(image == 254, pad_width=1, mode='constant', constant_values=0).astype(np.uint8)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
# eroded = cv2.erode(valid_space, kernel, iterations=1)

# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# dilated = cv2.dilate(eroded, kernel2, iterations=1)

# skeleton = skeletonize(eroded)
# skeleton_og = skeletonize(valid_space)

# fig = plt.figure(figsize=(10, 5))
# fig.add_subplot(1, 4, 1)
# plt.imshow(valid_space, cmap='gray')
# fig.add_subplot(1, 4, 2)
# plt.imshow(eroded, cmap='gray')
# fig.add_subplot(1, 4, 3)
# plt.imshow(dilated, cmap='gray')
# fig.add_subplot(1, 4, 4)
# plt.imshow(skeleton, cmap='gray')
# fig.add_subplot(2, 4, 1)
# plt.imshow(skeleton_og, cmap='gray')
# plt.show()

def get_space(map):
    wierd = (map == 255).astype(np.uint8)
    image = map - wierd
    # print(image.shape)

    valid_space = image[1:-1, 1:-1] == 254
    valid_space = np.pad(valid_space, pad_width=1, mode='constant', constant_values=0).astype(np.uint8)
    

    return valid_space

def skeletonize_map(map):
    valid_space = get_space(map)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
    eroded = cv2.erode(valid_space, kernel, iterations=1)

    skeleton = skeletonize(eroded).astype(np.int8)

    return skeleton

# print(image.shape)
skeleton = skeletonize_map(image)

def choose_points(image):
    # mask = np.zeros((image.shape[0], image.shape[1], 3))
    # points = np.where(image == 1)
    # mask[points] = [255, 0, 0]
    potential_points = np.argwhere(image == 1)
    points = []
    mask = np.zeros(image.shape)
    counter = 0
    for i, c in enumerate(potential_points):
        # print(c)
        px = image[c[0], c[1]]
        if px == 1:
            if counter == 0:
                mask[c[0], c[1]] = 1
                points.append(c)
                counter = 10
            else:
                counter -= 1
        # print(counter)
    return points, mask

points, mask = choose_points(skeleton)

# image_og = np.pad(image, pad_width=1, constant_values=1)
# print(np.min(image), np.min(skeleton), np.min(mask))
# print(image_og.shape, skeleton.shape, mask.shape)

# image_i = np.transpose(np.array([image_og, image_og, image_og]), (1, 2, 0))
image_i = image_og
skeleton_i = np.transpose(np.array([skeleton, skeleton, skeleton]), (1, 2, 0)).astype(np.int8)*255
mask_i = np.transpose(np.array([mask, np.zeros(mask.shape), np.zeros(mask.shape)]), (1, 2, 0)).astype(np.int8)*255

merge = image_i - skeleton_i + mask_i
# print(points)
# plt.imshow(merge)
# plt.show()


def point_tf(y, x):
    shape = (153, 124)
    x = x / (shape[1]-1)
    y = y / (shape[0]-1)

    UL = [-2.16, 5.05, 0]
    UR = [4.03, 5.05, 0]
    DL = [-2.16, -2.59, 0]
    DR = [4.03, -2.59, 0]

    LR_len = UR[0] - UL[0]
    UD_len = DL[1] - UL[1]

    U = 0
    L = np.pi/2
    D = np.pi
    R = 3*np.pi/2
    x = x * LR_len + UL[0]
    y = y * UD_len + UL[1]
    
    return x, y

def inv_point_tf(x, y):
    shape = (6.19, -7.64)
    x = (x+2.16) / (shape[0])
    y = (y-5.05) / (shape[1])

    UL = [0, 0, 0]
    UR = [0, 123, 0]
    DL = [152, 0, 0]
    DR = [152, 123, 0]

    UD_len = 152
    LR_len = 123

    U = 0
    L = np.pi/2
    D = np.pi
    R = 3*np.pi/2
    x = x * LR_len
    y = y * UD_len
    
    return int(y), int(x)


# p00 = point_tf(0, 0)
# p01 = point_tf(0, 123)
# p10 = point_tf(152, 0)
# p11 = point_tf(152, 123)

# ip00 = inv_point_tf(-2.16, 5.05)
# ip01 = inv_point_tf(4.03, 5.05)
# ip10 = inv_point_tf(-2.16, -2.59)
# ip11 = inv_point_tf(4.03, -2.59)

# print(p00, p01, p10, p11)
# print(ip00, ip01, ip10, ip11)

def closest_man(array, origin=(0, 0)):
    closest_i = 0
    closest_dist = float("inf")
    for i, (y, x) in enumerate(array):
        i_dist = abs(y - origin[0]) + abs(x - origin[1])
        if i_dist <= closest_dist:
            closest_i = i
            closest_dist = i_dist
        
    return array[closest_i]

def find_start(skeleton, start=(0, 0)):
    start_point = inv_point_tf(*start)
    if skeleton[start_point[0], start_point[1]] == 1:
        return start_point
    found_candidate = False
    padding = 1
    while not found_candidate:
        candidates = skeleton[start_point[0]-padding:start_point[0]+padding+1, start_point[1]-padding:start_point[1]+padding+1]

        if np.any(candidates == 1):
            arg_candidates = np.transpose(np.nonzero(candidates == 1))
            point = closest_man(arg_candidates, (padding, padding))
            # print(point, start_point, padding)
            point[0] += start_point[0]-padding
            point[1] += start_point[1]-padding
            found_candidate = True
            # print(point, skeleton[point[0], point[1]])
        else:
            padding += 1
    
    return point

def find_next(skeleton, start=(0, 0), rem=False):
    start_point = start
    found_candidates = False
    padding = 1
    if skeleton[start_point[0], start_point[1]] == 1:
        skeleton[start_point[0], start_point[1]] = 0
    while not found_candidates:
        candidates = skeleton[start_point[0]-padding:start_point[0]+padding+1, start_point[1]-padding:start_point[1]+padding+1]
        if np.any(candidates == 1):
            arg_candidates = np.transpose(np.nonzero(candidates == 1)).T
            arg_candidates[0] += start_point[0]-padding
            arg_candidates[1] += start_point[1]-padding
            arg_candidates = arg_candidates.T
            found_candidates = True
        else:
            padding += 1

    return arg_candidates, skeleton


def create_path(skeleton, start=(0,0)):
    prev = skeleton.copy()
    skeleton = skeleton
    start = find_start(skeleton)
    print("Start", start)
    path = [start]
    next_candidates, skeleton = find_next(skeleton, start=start)
    next_candidates = [next_candidates[-1]]
    print(next_candidates)
    # plt.imshow(skeleton)
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.close()
    while len(np.transpose(np.nonzero(skeleton == 1))) != 0:
        print(next_candidates)
        if len(next_candidates) == 1:
            next_candidates, skeleton = find_next(skeleton, start=next_candidates[0])
        else:
            next_candidates, skeleton = find_next(skeleton, start=next_candidates[0])
        plt.imshow(skeleton)
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()
        # skeleton = skeleton*0
        if next_candidates[0][0] < 0:
            break

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(skeleton)
    fig.add_subplot(1, 2, 2)
    plt.imshow(prev)
    plt.show()
    return path

# print(image.shape)
import time
stime = time.time()
path = create_path(skeleton)
print("Execution time:", (time.time()-stime)*len(skeleton == True))

def choze():
    ...
