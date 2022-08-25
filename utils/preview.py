import numpy as np
import matplotlib.pyplot as plt
import cv2

LUT_MASK_V1 = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5], [0, 0, 0]])
LUT_MASK_V2 = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])

def save_preview_input_gt_output(filename, input_image, gt, output):
    fig, axarr = plt.subplots(1, 3)
    fig.set_figheight(3)
    fig.set_figwidth(8)
    #axarr[0].set_title("Input")
    axarr[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    #axarr[1].set_title("GT")
    axarr[1].imshow(gt)
    #axarr[2].set_title("Output")
    axarr[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axarr[2].imshow(output)
    plt.savefig(filename)
    plt.close(fig)


def preview_img_and_mask(img, mask):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    plt.show()


def preview_patch(img):
    plt.imshow(img)
    plt.show()


def apply_lut_mask(mask, lut=LUT_MASK_V2):
    # Classes: 1 (Asbestos), 2 (Buildings), 3 (Streets), 4 (Greenspaces), 5 (Others)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3));
    mask_rgb[:, :, 0] = lut[:, 0][mask]
    mask_rgb[:, :, 1] = lut[:, 1][mask]
    mask_rgb[:, :, 2] = lut[:, 2][mask]
    return mask_rgb


def preview_mask(mask):
    plt.imshow(apply_lut_mask(mask), cmap='gray')
    plt.show()


def preview_mask_category(mask, category):
    plt.imshow(mask == category, cmap='gray')
    plt.show()
