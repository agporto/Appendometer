# Part of the standard library
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import re
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
    part.set("name", str(int(id)))
    part.set("x", str(int(x)))
    part.set("y", str(int(y)))

    return part


def pretty_xml(elem, out):
    """
    Writes the xml file to disk

    Parameters:
    ----------
        elem (Element): root element
        out (str): name of the output file

    Returns:
    ----------
        None (xml file written to disk)
    """
    et = ET.ElementTree(elem)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out, "w") as f:
        f.write(xmlstr)


def predictions_to_xml(
    predictor_name: str, folder: str, ignore=None, output="output.xml", max_error=None
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
    files = glob.glob(f"./{folder}/*")

    predictor = dlib.shape_predictor(predictor_name)

    error_root, error_images_e = initialize_xml()
    accurate_root, accurate_images_e = initialize_xml()

    kernel = np.ones((7, 7), np.float32) / 49

    print_error = False

    for f in sorted(files, key=str):
        ext = ntpath.splitext(f)[1]
        if ext.lower() in extensions:
            print(f"Processing image {f}")
            image_e = ET.Element("image")
            image_e.set("file", str(f))
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w = img.shape[1]
            h = img.shape[0]
            landmarks = []
            for scale in scales:
                image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                image = cv2.filter2D(image, -1, kernel)
                image = cv2.bilateralFilter(image, 9, 41, 21)

                rect = dlib.rectangle(1, 1, int(w * scale) - 1, int(h * scale) - 1)
                shape = predictor(image, rect)
                landmarks.append(shape_to_np(shape) / scale)

            box = create_box(img.shape)
            part_length = range(0, shape.num_parts)
            count = 0
            for item, i in enumerate(sorted(part_length, key=str)):
                x = np.median([landmark[item][0] for landmark in landmarks])
                y = np.median([landmark[item][1] for landmark in landmarks])
                if ignore is not None:
                    if i not in ignore:
                        part = create_part(x, y, i)
                        box.append(part)
                else:
                    part = create_part(x, y, i)
                    box.append(part)

                pos = np.array(landmarks)[:, item]
                pos_x, pos_y = (
                    pos[:, 0],
                    pos[:, 1],
                )

                mean_x, mean_y = np.mean(pos_x), np.mean(pos_y)
                distances = np.sqrt((pos_x - mean_x) ** 2 + (pos_y - mean_y) ** 2)
                total_variance = np.mean(distances)

                if max_error is not None:
                    if total_variance > max_error:
                        print(f"High error landmark: {item}")
                        print(f"Error: {total_variance}")
                        count += 1

                else:
                    pass

        box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
        image_e.append(box)

        if count > 1:
            print_error = True
            error_images_e.append(image_e)

        else:
            accurate_images_e.append(image_e)

    # Write the xml files to disk
    if max_error is None:
        pretty_xml(accurate_root, output)
    else:
        if print_error:
            pretty_xml(error_root, "error_" + output)
        pretty_xml(accurate_root, "accurate_" + output)


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


def natural_sort(l):
    """
    Internal function used by the dlib_xml_to_pandas. Performs the natural sorting of an array of XY
    coordinate names.

    Parameters:
    ----------
        l(array)=array to be sorted

    Returns:natural_sort_XY
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
    df = df[natural_sort(df)]
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
