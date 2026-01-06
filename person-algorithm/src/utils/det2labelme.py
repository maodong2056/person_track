from PIL import Image
import base64
import os
flags = {
    "bbox": False,
    "fbox": False,
    "left_eye": False,
    "right_eye": False,
    "nose": False,
    "left_mouth": False,
    "right_mouth": False,
    "lhbox": False,
    "lh_1": False,
    "lh_2": False,
    "lh_3": False,
    "lh_4": False,
    "rhbox": False,
    "rh_1": False,
    "rh_2": False,
    "rh_3": False,
    "rh_4": False,
}

def detection_to_labelme(filepath:str, results:list):
    output = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageData": "",
        "imageHeight": 0,
        "imageWidth": 0
    }
    (filepath_root, tempfilename) = os.path.split(filepath)
    with open(filepath, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    img = Image.open(filepath)
    width, height = img.size
    output["imagePath"] = tempfilename
    output["imageData"] = imageData
    output["imageHeight"] = height
    output["imageWidth"] = width
    label_id = 0 ## class human label count
    for res in results:
        label = str(label_id) if "id" not in res.keys() else res["id"]
        label_id += 1
        bbox = res["body_box"]
        flag = flags.copy()
        flag["bbox"] = True # The body box
        output["shapes"].append({
            "label": label,
            "points": [
                [float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": flag
        })
        fbox = res["face_box"]
        landmark = res["face_lm"]
        if fbox is not None:
            flag = flags.copy()
            flag["fbox"] = True  # The face box
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(fbox[0]), float(fbox[1])],
                    [float(fbox[2]), float(fbox[3])]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": flag
            })
            flag = flags.copy()
            flag["left_eye"] = True  # The face landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(landmark[0]), float(landmark[1])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["right_eye"] = True  # The face landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(landmark[2]), float(landmark[3])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["nose"] = True  # The face landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(landmark[4]), float(landmark[5])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["left_mouth"] = True  # The face landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(landmark[6]), float(landmark[7])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["right_mouth"] = True  # The face landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(landmark[8]), float(landmark[9])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
        lh_box = res["lh_box"]
        lh_landmark = res["lh_lm"]
        if lh_box is not None:
            flag = flags.copy()
            flag["lhbox"] = True  # The lh box
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(lh_box[0]), float(lh_box[1])],
                    [float(lh_box[2]), float(lh_box[3])]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": flag
            })
            flag = flags.copy()
            flag["lh_1"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(lh_landmark[0]), float(lh_landmark[1])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["lh_2"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(lh_landmark[2]), float(lh_landmark[3])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["lh_3"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(lh_landmark[4]), float(lh_landmark[5])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["lh_4"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(lh_landmark[6]), float(lh_landmark[7])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })

        rh_box = res["rh_box"]
        rh_landmark = res["rh_lm"]
        if rh_box is not None:
            flag = flags.copy()
            flag["rhbox"] = True  # The lh box
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(rh_box[0]), float(rh_box[1])],
                    [float(rh_box[2]), float(rh_box[3])]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": flag
            })
            flag = flags.copy()
            flag["rh_1"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(rh_landmark[0]), float(rh_landmark[1])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["rh_2"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(rh_landmark[2]), float(rh_landmark[3])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["rh_3"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(rh_landmark[4]), float(rh_landmark[5])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
            flag = flags.copy()
            flag["rh_4"] = True  # The lh landmark
            output["shapes"].append({
                "label": label,
                "points": [
                    [float(rh_landmark[6]), float(rh_landmark[7])],
                ],
                "group_id": None,
                "shape_type": "point",
                "flags": flag
            })
    return output
