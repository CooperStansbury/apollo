from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path
import re
import os
import sys
from tabulate import tabulate


BASE_DIR = Path(workflow.basedir)
configfile: str(BASE_DIR) + "/config/config.yaml"

# big picture variables
OUTPUT = config['output_path']
ext = config['extension']

# load in images paths
input_path = os.path.abspath(config['inputs'])
input_df = pd.read_csv(input_path, comment="#")
samples = input_df['sample_id'].to_list()

# get new path names
output_paths = [OUTPUT + "images/" + x + ext for x in samples]

# print statements
print("\n----- CONFIG VALUES -----")
for key, value in config.items():
    print(f"{key}: {value}")
    
    
print("\n----- INPUT VALUES -----")
print(
    tabulate(
        input_df, 
        headers='keys', 
        tablefmt='psql',
        showindex=False,
    )
)

rule all:
    input:
        expand(OUTPUT + "images/{sid}.ome.tiff", sid=samples),
        expand(OUTPUT + "channel_intensities/{sid}.mean_intensity.csv", sid=samples),
        expand(OUTPUT + "pixel_counts/{sid}.csv", sid=samples),
        expand(OUTPUT + "gif/{sid}.gif", sid=samples),
        
   
rule get_images:
    input:
        input_df['file_path'].to_list()
    output:
        output_paths
    run:
        from shutil import copyfile
        for i, refPath in enumerate(input):

            outPath = output[i]
            copyfile(refPath, outPath)
            
                        
rule get_mean_intensities:
    input:
        OUTPUT + "images/{sid}.ome.tiff",
    output:
        OUTPUT + "channel_intensities/{sid}.mean_intensity.csv",
    conda:
        "imaging"
    wildcard_constraints:
        sid='|'.join([re.escape(x) for x in set(samples)]),
    shell:
        """python scripts/mean_intensity.py {input} {output}"""
        
        
rule segment:
    input:
        OUTPUT + "images/{sid}.ome.tiff",
    output:
        seg=OUTPUT + "segmentation/{sid}.tiff",
        props=OUTPUT + "region_props/{sid}.csv",
    conda:
        "stardist"
    wildcard_constraints:
        sid='|'.join([re.escape(x) for x in set(samples)]),
    params:
        model=config['stardist_model'],
        prob_thresh=config['prob_thresh'],
        nms_thresh=config['nms_thresh'],
    shell:
        """python scripts/segment.py {input} \
        {params.model} {params.prob_thresh} \
        {params.nms_thresh} {output.seg} \
        {output.props}"""
        
        
rule label_cells:
    input:
        OUTPUT + "segmentation/{sid}.tiff",
    output:
        OUTPUT + "labels/{sid}.tiff",
    conda:
        "imaging"
    wildcard_constraints:
        sid='|'.join([re.escape(x) for x in set(samples)]),
    shell:
        """python scripts/label_segmentation.py {input} {output}"""
        
        
        
rule count_pixels:
    input:
        OUTPUT + "labels/{sid}.tiff",
    output:
        OUTPUT + "pixel_counts/{sid}.csv",
    conda:
        "imaging"
    wildcard_constraints:
        sid='|'.join([re.escape(x) for x in set(samples)]),
    shell:
        """python scripts/count_pixels.py {input} {output}"""
        
        
rule make_movie:
    input:
        OUTPUT + "labels/{sid}.tiff",
    output:
        OUTPUT + "gif/{sid}.gif"
    conda:
        "imaging"
    wildcard_constraints:
        sid='|'.join([re.escape(x) for x in set(samples)]),
    shell:
        """python scripts/make_movie.py {input} {output}"""
    