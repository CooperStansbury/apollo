import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tifffile import imread
from tifffile import imwrite


def label_frame(red, green):
    """
    Combines two boolean images into a matrix with specific values based on overlap.

    Args:
        red: A boolean numpy array representing the red image.
        green: A boolean numpy array representing the green image.

    Returns:
        A numpy array with the same shape as the input images, containing the combined values.
    """

    # Create an empty output matrix of the same shape
    output = np.zeros_like(red, dtype=np.int32)

    # Assign values based on conditions
    output[(red == True) & (green == False)] = 1  # Red only
    output[(red == False) & (green == True)] = 2  # Green only
    output[(red == True) & (green == True)] = 3   # Both

    return output


def label_cells(img):
    """
    Labels cells in a multi-channel image based on red and green channels.

    Args:
        img: A 4D NumPy array representing the image data, 
             where the first dimension is the channel, the second is the time frame, 
             and the last two dimensions are the spatial dimensions (height and width).
             Assumes red channel is at index 2 and green channel is at index 3.

    Returns:
        A 3D NumPy array representing the labeled image, where each pixel
        is assigned a label based on its red and green channel values in the corresponding frame.
        The labels are determined by the 'label_frame' function.
    """

    label_image = np.zeros(img.shape[1:])

    red_channel = 2
    green_channel = 3

    for t in range(img.shape[1]):
        red = img[red_channel, t, :, :].astype(bool)
        green = img[green_channel, t, :, :].astype(bool)
        label = label_frame(red, green)
        label_image[t, :, :] = label

    return label_image

    
    
    
if __name__ == "__main__":
    img_path = sys.argv[1]
    output_path = sys.argv[2]

    img = imread(img_path)
    
    label_img = label_cells(img)

    # save labelling
    imwrite(output_path, label_img)
    print(f"Labels saved to: {output_path}") 
    
    
    
