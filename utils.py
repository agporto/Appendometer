# Part of the standard library
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import csv
import re
import random
import shutil
import glob
import ntpath

# Not part of the standard library
import numpy as np
import pandas as pd
import cv2
import dlib


# Tools for predicting objects and shapes in new images


def initialize_xml():
    """
    Initializes the xml file for the predictions

    Parameters:
    ----------
        None

    Returns:
    ----------
        None (xml file written to disk)
    """
    root = ET.Element("dataset")
    root.append(ET.Element("name"))
    root.append(ET.Element("comment"))
    images_e = ET.Element("images")
    root.append(images_e)

    return root, images_e


def create_box(img_shape):
    """
    Creates a box around the image

    Parameters:
    ----------
        img_shape (tuple): shape of the image

    Returns:
    ----------
        box (Element): box element
    """
    box = ET.Element("box")
    box.set("top", str(int(1)))
    box.set("left", str(int(1)))
    box.set("width", str(int(img_shape[1] - 2)))
    box.set("height", str(int(img_shape[0] - 2)))

    return box


def create_part(x, y, id):
    """
    Creates a part element

    Parameters:
    ----------
        x (int): x coordinate of the part
        y (int): y coordinate of the part
        name (str): name of the part

    Returns:
    ----------
        part (Element): part element
    """
    part = ET.Element("part")
    part.set("name", int(id))
    part.set("x", str(int(x)))
    part.set("y", str(int(y)))

    return part


def predictions_to_xml_tta(
    predictor_name: str, dir: str, ignore=None, out_file="output.xml", var_threshold=43
):
    """
    Generates dlib format xml files for model predictions. It uses previously trained models to
    identify objects in images and to predict their shape.

    Parameters:
    ----------
        predictor_name (str): shape predictor filename
        dir(str): name of the directory containing images to be predicted
        ratio (float): (optional) scaling factor for the image
        out_file (str): name of the output file (xml format)
        variance_threshold (float): threshold value to determine high variance images

    Returns:
    ----------
        None (out_file written to disk)
    """
    extensions = {".jpg", ".jpeg", ".tif", ".png", ".bmp"}
    scales = [0.25, 0.5, 1]

    predictor = dlib.shape_predictor(predictor_name)

    high_var_root, high_var_images_e = initialize_xml()
    low_var_root, low_var_images_e = initialize_xml()

    kernel = np.ones((7, 7), np.float32) / 49

    image_count = 0
    for f in glob.glob(f"./{dir}/*"):
        ext = ntpath.splitext(f)[1]
        if ext.lower() in extensions:
            image_count += 1
            print(f"Processing image {image_count}")
            image_e = ET.Element("image")
            image_e.set("file", str(f))
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply test time augmentation on the image

            all_preds = []
            for scale in scales:
                # Resize the image
                image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                image = cv2.filter2D(image, -1, kernel)
                image = cv2.bilateralFilter(image, 9, 41, 21)

                e = dlib.rectangle(
                    left=1,
                    top=1,
                    right=int(image.shape[1]) - 1,
                    bottom=int(image.shape[0]) - 1,
                )

                shape = predictor(image, e)
                all_preds.append(shape_to_np(shape) * 1 / scale)

            box = create_box(img.shape)
            part_length = range(0, shape.num_parts)
            count = 0
            for item, i in enumerate(sorted(part_length, key=str)):
                x = np.median([pred[item][0] for pred in all_preds])
                y = np.median([pred[item][1] for pred in all_preds])
                if ignore is not None:
                    if i not in ignore:
                        part = create_part(x, y, i)
                        box.append(part)
                else:
                    part = create_part(x, y, i)
                    box.append(part)

                # Calculate variance for each part x and y position
                positions = np.array(all_preds)[:, item]  # select positions for the landmark
                positions_x, positions_y = (positions[:, 0], positions[:, 1])  # separate x and y positions

                # Step 1: Compute the mean X and Y positions of the landmark.
                mean_x, mean_y = np.mean(positions_x), np.mean(positions_y)

                # Step 2: Compute the Euclidean distance between each observation's X and Y positions and the mean X and Y positions.
                distances = np.sqrt(
                    (positions_x - mean_x) ** 2 + (positions_y - mean_y) ** 2
                )

                # Step 3: Take the average of the Euclidean distances to get the average Euclidean distance from the mean.
                total_variance = np.mean(distances)

                # Check if the variance is greater than the threshold and add the image to the corresponding xml
                if total_variance > var_threshold:
                    print(f"High variance image: {f}")
                    print(f"Variance: {total_variance}")

                    count += 1
                else:
                    pass
        box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
        image_e.append(box)
        if count > 1:
            high_var_images_e.append(image_e)
        else:
            low_var_images_e.append(image_e)

    # Write the xml files to disk
    high_var_et = ET.ElementTree(high_var_root)
    high_var_xmlstr = minidom.parseString(
        ET.tostring(high_var_et.getroot())
    ).toprettyxml(indent="   ")
    with open("high_var_" + out_file, "w") as f:
        f.write(high_var_xmlstr)

    low_var_et = ET.ElementTree(low_var_root)
    low_var_xmlstr = minidom.parseString(ET.tostring(low_var_et.getroot())).toprettyxml(
        indent="   "
    )
    with open("low_var_" + out_file, "w") as f:
        f.write(low_var_xmlstr)


def shape_to_np(shape):
    """
    Convert a dlib shape object to a NumPy array of (x, y)-coordinates.

    Parameters
    ----------
    shape : dlib.full_object_detection
        The dlib shape object to convert.

    Returns
    -------
    coords: np.ndarray
        A NumPy array of (x, y)-coordinates representing the landmarks in the input shape object.
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype="int")

    length = range(0, shape.num_parts)

    for i in sorted(length, key=str):
        coords[i] = [shape.part(i).x, shape.part(i).y]

    # return the list of (x, y)-coordinates
    return coords


# Importing to pandas tools


def natural_sort_XY(l):
    """
    Internal function used by the dlib_xml_to_pandas. Performs the natural sorting of an array of XY
    coordinate names.

    Parameters:
    ----------
        l(array)=array to be sorted

    Returns:
    ----------
        l(array): naturally sorted array
    """
    convert = lambda text: int(text) if text.isdigit() else 0
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def dlib_xml_to_pandas(xml_file: str, parse=False):
    """
    Imports dlib xml data into a pandas dataframe. An optional file parsing argument is present
    for very specific applications. For most people, the parsing argument should remain as 'False'.

    Parameters:
    ----------
        xml_file(str):file to be imported (dlib xml format)

    Returns:
    ----------
        df(dataframe): returns a pandas dataframe containing the data in the xml_file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    landmark_list = []
    for images in root:
        for image in images:
            for boxes in image:
                box = (
                    boxes.attrib["top"]
                    + "_"
                    + boxes.attrib["left"]
                    + "_"
                    + boxes.attrib["width"]
                    + "_"
                    + boxes.attrib["height"]
                )
                for parts in boxes:
                    if parts.attrib["name"] is not None:
                        data = {
                            "id": image.attrib["file"],
                            "box_id": box,
                            "box_top": float(boxes.attrib["top"]),
                            "box_left": float(boxes.attrib["left"]),
                            "box_width": float(boxes.attrib["width"]),
                            "box_height": float(boxes.attrib["height"]),
                            "X" + parts.attrib["name"]: float(parts.attrib["x"]),
                            "Y" + parts.attrib["name"]: float(parts.attrib["y"]),
                        }

                    landmark_list.append(data)
    dataset = pd.DataFrame(landmark_list)
    df = dataset.groupby(["id", "box_id"], sort=False).max()
    df = df[natural_sort_XY(df)]
    basename = ntpath.splitext(xml_file)[0]
    df.to_csv(f"{basename}.csv")
    return df


def dlib_xml_to_tps(xml_file: str):
    """
    Imports dlib xml data and converts it to tps format

    Parameters:
    ----------
        xml_file(str):file to be imported (dlib xml format)

    Returns:
    ----------
        tps (file): returns the dataset in tps format
    """
    basename = ntpath.splitext(xml_file)[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    id = 0
    coordinates = []
    with open(f"{basename}.tps", "w") as f:
        wr = csv.writer(f, delimiter=" ")
        for images in root:
            for image in images:
                for boxes in image:
                    wr.writerows([["LM=" + str(int(len(boxes)))]])
                    for parts in boxes:
                        if parts.attrib["name"] is not None:
                            data = [
                                float(parts.attrib["x"]),
                                float(boxes.attrib["height"])
                                + 2
                                - float(parts.attrib["y"]),
                            ]
                        wr.writerows([data])
                    wr.writerows(
                        [["IMAGE=" + image.attrib["file"]], ["ID=" + str(int(id))]]
                    )
                    id += 1
