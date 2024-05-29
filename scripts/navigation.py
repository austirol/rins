import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

image = cv2.imread(os.path.join(os.getcwd(), "..", "maps", "mapademo3.pgm"), cv2.IMREAD_GRAYSCALE)
image_og = cv2.imread(os.path.join(os.getcwd(), "..", "maps", "mapademo3.pgm"), cv2.IMREAD_COLOR)

def get_space(map):
    # print(map.shape)    
    wierd = (map == 255)
    # print(wierd)
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
    
    return [x, y]

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

# point00 = inv_point_tf(0, 0)
# print("00", point00)

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

def find_start(skeleton, start):
    start_point = start
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

def get_points(map, spacing=15):

    skeleton = skeletonize_map(map)

    start_point = find_start(skeleton, inv_point_tf(0, 0))
    
    points = [start_point]
    current_point = start_point
    current_distance = 0

    # Find skeleton coordinates
    skeleton_indices = np.argwhere(skeleton)
    while len(skeleton_indices) > 0:
        # Calculate distances to skeleton points

        distances = np.linalg.norm(skeleton_indices - current_point, axis=1)

        min_index = np.argmin(distances)
        nearest_point = skeleton_indices[min_index]
        nearest_distance = distances[min_index]

        # Move to the nearest skeleton point
        current_point = tuple(nearest_point)
        current_distance += nearest_distance

        # If current distance exceeds spacing, add the point to the list
        if current_distance >= spacing:
            points.append(current_point)
            current_distance = 0

        # Remove the used point from skeleton_indices
        skeleton_indices = np.delete(skeleton_indices, min_index, axis=0)

    out = []
    for y, x in points:
        po = point_tf(x, y)
        po = [po[1], po[0]]
        po.append(((np.random.rand(1)-0.5)*np.pi)[0])
        out.append(po)
        print(po)
    points = out

    return points


def draw_path(image, points):
    skeli = skeletonize_map(image)
    image = np.transpose(np.array([image, image, image]), (1, 2, 0))
    skeli = np.transpose(np.array([skeli, skeli, skeli]), (1, 2, 0)).astype(np.int8)*255
    mask = np.zeros_like(image)
    # print(mask.shape)
    for i in points:
        x, y= inv_point_tf(i[0], i[1])
        print(y, x)
        # x = inv_point_tf()
        # print(type(mask))
        mask[y, x] = np.array([1, 0, 0])
        plt.imshow(mask)
        # plt.show()
        # plt.show(block=False)
        # plt.pause(0.00001)
        # plt.close()
    mask_i = mask.astype(np.int8)*255
    merge = image - skeli + mask_i

    plt.imshow(merge)
    plt.show()

if __name__ == "__main__":
    path = get_points(image, 16)
    print(path)
    print(inv_point_tf(path[0][0], path[0][1]))
    draw_path(image[:,:], path)
