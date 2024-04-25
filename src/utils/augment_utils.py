import cv2
import numpy as np
import random



def scale_image(img, scale):
    """ Scale an image by a given factor using OpenCV. """
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def check_overlap(new_rect, rects):
    """ Check if the new rectangle overlaps with any existing ones. """
    x_new, y_new, w_new, h_new = new_rect
    for rect in rects:
        x, y, w, h = rect
        if not (x_new + w_new <= x or x_new >= x + w or y_new + h_new <= y or y_new >= y + h):
            
            return True
    return False

def random_scale(img, scale_low, scale_high):
    """ Randomly scale an image within the specified scale range using OpenCV. """
    scale = random.uniform(scale_low, scale_high)
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def place_images(img_a, img_b, scale_range):
    """ Place as many randomly scaled non-overlapping copies of img_a onto img_b as possible. """
    h_b, w_b, _ = img_b.shape
    rectangles = []
    viz_img = img_b.copy()

    while True:
        scaled_img_a = random_scale(img_a, *scale_range)
        h_a, w_a, _ = scaled_img_a.shape
        
        # Attempt to place the image randomly up to 100 times per scale
        for _ in range(100):
            x = random.randint(0, w_b - w_a)
            y = random.randint(0, h_b - h_a)
            if not check_overlap((x, y, w_a, h_a), rectangles):
                # No overlap, place the image
                viz_img[y:y+h_a, x:x+w_a] = scaled_img_a
                rectangles.append((x, y, w_a, h_a))
                break
        else:
            break

    return viz_img


def place_fixed_scale_images_with_shift(img_a, img_b, obj_scale, shift_x, shift_y):
    scaled_img_a = scale_image(img_a, obj_scale)
    h_a, w_a, _ = scaled_img_a.shape
    h_b, w_b, _ = img_b.shape

    viz_img = np.zeros_like(img_b)

    # Calculate the number of images that fit into img_b
    num_x = w_b // w_a
    num_y = h_b // h_a

    # Place the images with the given shift
    for ix in range(num_x):
        for iy in range(num_y):
            # Calculate the top-left corner position
            pos_x = (ix * w_a + shift_x) % w_b
            pos_y = (iy * h_a + shift_y) % h_b

            # Select the region in the visualization image where we'll place the scaled_img_a
            slice_x = slice(pos_x, pos_x + w_a)
            slice_y = slice(pos_y, pos_y + h_a)

            # Handle wrapping around for both x and y axes
            if pos_x + w_a > w_b:
                continue 

            elif pos_y + h_a > h_b:
                continue

            else:
                viz_img[slice_y, slice_x] = scaled_img_a

    return viz_img

