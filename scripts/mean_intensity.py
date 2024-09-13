import pandas as pd
import numpy as np
import os
import sys
from tifffile import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.exposure import rescale_intensity


def get_channel_means(img):
    """
    Computes the mean intensity of all channels over time.

    Args:
        img (numpy.ndarray): The image data with shape (num_channels, time, height, width, 3).

    Returns:
        pd.DataFrame: A DataFrame with columns for each channel and 'time', containing the mean intensities over time.
    """

    num_channels, time_steps, _, _, _ = img.shape
    column_names = [f"Channel {i}" for i in range(num_channels)]

    # Preallocate the result array for efficiency
    result_array = np.zeros((time_steps, num_channels + 1)) 

    for t in range(time_steps):
        img_t = img[:, t, :, :, :]
        img_t = rgb2gray(img_t)
        img_t = rescale_intensity(img_t, out_range=(0, 255))

        means = np.mean(img_t, axis=(1, 2))
        result_array[t, :-1] = means
        result_array[t, -1] = t  # Store the time in the last column

    # Create the DataFrame directly from the result array
    result_df = pd.DataFrame(result_array, columns=column_names + ['Time'])

    return result_df

        
if __name__ == "__main__":
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    
    img = imread(img_path)
    
    result_df = get_channel_means(img)
    result_df.to_csv(out_path, index=False)

   

