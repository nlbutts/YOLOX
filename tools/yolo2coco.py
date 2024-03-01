# This Python code converts a dataset in YOLO format into the COCO format.
# The YOLO dataset contains images of bottles and the bounding box annotations in the
# YOLO format. The COCO format is a widely used format for object detection datasets.

# The input and output directories are specified in the code. The categories for
# the COCO dataset are also defined, with only one category for "bottle". A dictionary for the COCO dataset is initialized with empty values for "info", "licenses", "images", and "annotations".

# The code then loops through each image in the input directory. The dimensions
# of the image are extracted and added to the COCO dataset as an "image" dictionary,
# including the file name and an ID. The bounding box annotations for each image are
# read from a text file with the same name as the image file, and the coordinates are
# converted to the COCO format. The annotations are added to the COCO dataset as an
# "annotation" dictionary, including an ID, image ID, category ID, bounding box coordinates,
# area, and an "iscrowd" flag.

# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
from PIL import Image
import glob
from pathlib import Path

# Set the paths for the input and output directories
input_dir = '/home/nlbutts/projects/YOLOX/Canola/Run_2'
output_dir = '/home/nlbutts/projects/YOLOX/coco'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Define the categories for the COCO dataset
categories = [{'id': 0, 'name': 'seed'}]

# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

image_files = glob.glob(f'{input_dir}/images/*.bmp')

# Loop through the images in the input directory
for image_file in image_files:

    # Load the image and get its dimensions
    image = Image.open(image_file)
    width, height = image.size

    # Add the image to the COCO dataset
    id = int(image_file.split('/')[-1].split('.')[0])
    image_dict = {
        "id": id,
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)

    # Load the bounding box annotations for the image
    labelfile = f'{input_dir}/labels/{id}.txt'
    with open(labelfile, 'r') as f:
        annotations = f.readlines()

    # Loop through the annotations and add them to the COCO dataset
    for ann in annotations:
        x, y, w, h = map(float, ann.strip().split()[1:])
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        ann_dict = {
            "id": len(coco_dataset["annotations"]),
            "image_id": id,
            "category_id": 0,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        }
        coco_dataset["annotations"].append(ann_dict)

# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(coco_dataset, f)
