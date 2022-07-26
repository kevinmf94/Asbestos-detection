import cv2
from PIL import Image
import numpy as np

RESIZE_ENABLED = True
RESIZE = (256, 256)
OTHERS_CLASS = 5


def postprocessmask_and_save(image, outfile, resize=RESIZE, resize_enable=RESIZE_ENABLED):
    image[image == 127] = OTHERS_CLASS
    image[image == 0] = OTHERS_CLASS
    if resize_enable:
        image = cv2.resize(image, RESIZE, interpolation=cv2.INTER_NEAREST)

    Image.fromarray(image).save(outfile)


def postprocess_and_save(image, outfile, resize_enable=RESIZE_ENABLED):
    if resize_enable:
        image = cv2.resize(image, RESIZE)

    Image.fromarray(image).save(outfile)
