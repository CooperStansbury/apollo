import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tifffile import imread
from tifffile import imwrite
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from csbdeep.utils import normalize
from stardist.models import StarDist2D



def segment_image(img, stardist_model, prob_thresh, nms_thresh):
    """
    Segments cells in an image using StarDist and extracts region properties.

    Args:
        img (numpy.ndarray): The image data with shape (num_channels, time, height, width, 3).
        stardist_model (str): Name of the pretrained StarDist model to use.
        prob_thresh (float): Probability threshold for object detection.
        nms_thresh (float): Non-maximum suppression threshold for object detection.

    Returns:
        tuple: A tuple containing:
            - segments (numpy.ndarray): A labeled image with the same shape as the input image (except for the last dimension), 
                                       where each pixel's value represents the segment it belongs to.
            - region_props (pd.DataFrame): A DataFrame containing the extracted region properties for each detected segment,
                                           including 'Channel' and 'Time' columns.
    """

    region_props = []
    segments = np.zeros((img.shape[:-1]))  # Initialize the segments array

    # Load the StarDist model
    model = StarDist2D.from_pretrained(stardist_model)

    for c in range(img.shape[0]):
        for t in range(img.shape[1]):
            # Extract the image at the current channel and time point, and convert to grayscale
            img_t = img[c, t, :, :, :]
            img_t = rgb2gray(img_t)
            
            # Predict instances (labels and other data) using the StarDist model
            try:
                labels, _ = model.predict_instances(
                    normalize(img_t),
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh
                )
            except:
                continue
                
            print(f"Channel: {c} Time: {t} Detected: {len(np.unique(labels))} cells")

            # Store the labels in the segments array
            segments[c, t, :, :] = labels

            # Extract region properties using regionprops_table
            props = regionprops_table(
                labels,
                intensity_image=img_t,
                properties=[
                    'label', 'area', 'bbox_area', 'convex_area', 'eccentricity',
                    'equivalent_diameter', 'extent', 'filled_area', 'major_axis_length',
                    'minor_axis_length', 'orientation', 'perimeter', 'max_intensity',
                    'mean_intensity', 'min_intensity', 'centroid'
                ]
            )

            # Create a DataFrame from the properties, and add 'Channel' and 'Time' columns
            df = pd.DataFrame(props)
            df['Channel'] = c
            df['Time'] = t

            region_props.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    if not region_props:  
        region_props = pd.DataFrame(columns=[
                    'label', 'area', 'bbox_area', 'convex_area', 'eccentricity',
                    'equivalent_diameter', 'extent', 'filled_area', 'major_axis_length',
                    'minor_axis_length', 'orientation', 'perimeter', 'max_intensity',
                    'mean_intensity', 'min_intensity', 'centroid', 'Channel', 'Time',
                ]
            )
    else:
        region_props = pd.concat(region_props)
    
    return segments, region_props
    
    
    
if __name__ == "__main__":
    img_path = sys.argv[1]
    stardist_model = sys.argv[2]
    prob_thresh = float(sys.argv[3])
    nms_thresh = float(sys.argv[4])
    segment_out_path = sys.argv[5]
    props_out_path = sys.argv[6]

    # Print input parameters and their values
    print("Input parameters:")
    print(f"  Image path: {img_path}")
    print(f"  StarDist model: {stardist_model}")
    print(f"  Probability threshold: {prob_thresh}")
    print(f"  NMS threshold: {nms_thresh}")
    print(f"  Segmentation output path: {segment_out_path}")
    print(f"  Properties output path: {props_out_path}")

    print("Starting image segmentation...\n") 

    img = imread(img_path)

    segments, region_props = segment_image(
        img, 
        stardist_model,
        prob_thresh,
        nms_thresh
    )

    print("Segmentation complete!") 

    print(region_props)

    # save segmentation
    imwrite(segment_out_path, segments)
    print(f"Segmentation saved to: {segment_out_path}")

    # save props
    region_props.to_csv(props_out_path, index=False)
    print(f"Region properties saved to: {props_out_path}") 
    
    
    
