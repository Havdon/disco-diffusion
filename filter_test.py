from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_mask(img, bg_color = [0,0,0]):
    black_pixels_mask = np.all(img == bg_color, axis=-1)
    other_pixels = ~black_pixels_mask
    img[black_pixels_mask] = [0, 0, 0]
    img[other_pixels] = [255,255,255]

    kernel = np.ones((20,20), np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    mask_blur = cv2.GaussianBlur(dilation, (101, 101), 0)
    mask_blur[other_pixels] = [255,255,255]
    return mask_blur

img = np.array(Image.open('image.png'))
mask = get_mask(img)

plt.figure()
plt.imshow(mask) 
plt.show()  # display it

