import pandas as pd
import numpy as np
import os
import sys
from tifffile import imread


def count_pixels(img):
    """
    Counts the number of pixels for each unique value in each frame of a 3D numpy array 
    and returns the results in a Pandas DataFrame.

    Args:
        img: A 3D numpy array where the first dimension represents frames, 
             and each 2D frame contains values from a set of unique values.

    Returns:
        A Pandas DataFrame where each row represents a frame, 
        columns are the unique values, and cell values are the corresponding pixel counts.
    """

    all_counts = []
    for frame in img:
        unique_values, counts = np.unique(frame, return_counts=True)
        all_counts.append(dict(zip(unique_values, counts)))

    df = pd.DataFrame(all_counts).fillna(0)  # Create DataFrame and fill missing values with 0
    df = df.reset_index()
    df.columns = ['Time', 'None', 'Red', 'Green', 'Both']
    return df


if __name__ == "__main__":
    img_path = sys.argv[1]
    output_path = sys.argv[2]

    img = imread(img_path)
    
    df = count_pixels(img)

    # save counts
    df.to_csv(output_path, index=False)
    print(f"Labels saved to: {output_path}") 
    
    