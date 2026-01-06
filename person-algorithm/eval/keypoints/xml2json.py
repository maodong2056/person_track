import numpy as np
import xml.etree.ElementTree as ET
import os
import json
#-------------------------------- pose name -----------------------------------------------
pose_name = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
pose_name_rs = ["pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
                "left_knee", "left_ankle", "torso", "neck", "nose", "head",
                "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
                "right_elbow", "right_wrist"]
#------------------------------------------------------------------------------------------

# #--------------------------------- data dir ---------------------------------------------
# collect_dir = "/raid/hzj/data/keypoint_project/Collection/20210615"
# save_dir = os.path.join(collect_dir, "dataset.json")
# anns_dir = os.path.join(collect_dir, "4000xml")
# images_dir = os.path.join(collect_dir, "images")
# # json_dir = "json"
# labels = os.listdir(anns_dir)
# #----------------------------------------------------------------------------------------

def get_xml_to_json(ann_dir, images_dir):
    dic = {}
    dic['info'] = {
        "version": "1.0",
    }
    dic['licenses'] = [{
        "id": 1,
    }]
    dic['categories'] = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": pose_name,
        "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                     [8, 10],
                     [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    }]
    dic['images'] = []
    dic['annotations'] = []
    id_num = 0
    image_num = -1
    anns_dir = ann_dir
    images_dir = images_dir
    labels = os.listdir(anns_dir)
    for label_name in sorted(labels):  # 20201126160137_000135.xml
        image_num += 1
        if image_num % 100 == 0:
            print(image_num)

        image_name = label_name.replace(".xml", ".jpg")
        ann_file = os.path.join(anns_dir,
                                label_name)  # "/raid/hzj/data/keypoint_project/Collection/20210615/4000xml/20201126160137_000135.xml"
        image_file = os.path.join(images_dir, image_name)
        if not (os.path.exists(image_file) and os.path.exists(ann_file)):
            continue

        tree = ET.parse(ann_file)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        dic['images'].append({
            "license": 1,
            "file_name": image_name,
            "height": height,
            "width": width,
            "id": image_num
        })

        for obj in root.iter('object'):
            if obj.find("difficult").text == "1":
                continue
            # if obj.find("truncated").text == "1":
            # continue

            ann_dic = {}

            # person_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = (float(bbox.find('xmin').text))
            ymin = (float(bbox.find('ymin').text))
            xmax = (float(bbox.find('xmax').text))
            ymax = (float(bbox.find('ymax').text))
            w = xmax - xmin
            h = ymax - ymin

            ann_dic['bbox'] = [xmin, ymin, w, h]
            ann_dic['area'] = w * h
            ann_dic['image_id'] = image_num
            ann_dic['id'] = id_num
            ann_dic['category_id'] = 1
            ann_dic["segmentation"] = None
            ann_dic["iscrowd"] = 0

            keypoints = obj.find("keypoints")
            human_kps = []
            for p_name in pose_name:
                kps_name = keypoints.find(p_name)
                vis = int(kps_name.attrib["attribute"])
                kp = kps_name.text.split(",")
                x, y = float(kp[0]), float(kp[1])
                human_kps.append(int(x))
                human_kps.append(int(y))
                human_kps.append(vis)

            ann_dic['keypoints'] = human_kps
            ann_dic['num_keypoints'] = int(len(human_kps) / 3)

            id_num += 1
            dic['annotations'].append(ann_dic)
    return dic

def get_xml_to_json_rs(ann_dir, images_dir):
    dic = {}
    dic['info'] = {
        "version": "1.0",
    }
    dic['licenses'] = [{
        "id": 1,
    }]
    dic['categories'] = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": pose_name,
        "skeleton": [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                     [12, 13], [8, 14], [14, 15], [15, 16], [0, 1],
                     [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]
    }]
    dic['images'] = []
    dic['annotations'] = []
    id_num = 0
    image_num = -1
    anns_dir = ann_dir
    images_dir = images_dir
    labels = os.listdir(anns_dir)
    for label_name in sorted(labels):  # 20201126160137_000135.xml
        image_num += 1
        if image_num % 100 == 0:
            print(image_num)

        image_name = label_name.replace(".xml", ".jpg")
        ann_file = os.path.join(anns_dir,
                                label_name)  # "/raid/hzj/data/keypoint_project/Collection/20210615/4000xml/20201126160137_000135.xml"
        image_file = os.path.join(images_dir, image_name)
        if not (os.path.exists(image_file) and os.path.exists(ann_file)):
            continue

        tree = ET.parse(ann_file)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        dic['images'].append({
            "license": 1,
            "file_name": image_name,
            "height": height,
            "width": width,
            "id": image_num
        })

        for obj in root.iter('object'):
            if obj.find("difficult").text == "1":
                continue
            # if obj.find("truncated").text == "1":
            # continue

            ann_dic = {}

            # person_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = (float(bbox.find('xmin').text))
            ymin = (float(bbox.find('ymin').text))
            xmax = (float(bbox.find('xmax').text))
            ymax = (float(bbox.find('ymax').text))
            w = xmax - xmin
            h = ymax - ymin

            ann_dic['bbox'] = [xmin, ymin, w, h]
            ann_dic['area'] = w * h
            ann_dic['image_id'] = image_num
            ann_dic['id'] = id_num
            ann_dic['category_id'] = 1
            ann_dic["segmentation"] = None
            ann_dic["iscrowd"] = 0

            keypoints = obj.find("keypoints")
            human_kps = []
            for p_name in pose_name_rs:
                kps_name = keypoints.find(p_name)
                vis = int(kps_name.attrib["attribute"])
                if p_name == "nose":
                    face_keypoints = obj.find("face")
                    kps_name = face_keypoints.find(p_name)
                    vis = int(kps_name.attrib["attribute"])
                    if vis==0:
                        kps_name = keypoints.find(p_name)
                        vis = int(kps_name.attrib["attribute"])
                kp = kps_name.text.split(",")
                x, y = float(kp[0]), float(kp[1])
                human_kps.append(int(x))
                human_kps.append(int(y))
                human_kps.append(vis)
            ann_dic['keypoints'] = human_kps
            ann_dic['num_keypoints'] = int(len(human_kps) / 3)

            id_num += 1
            dic['annotations'].append(ann_dic)
    return dic

if __name__ == '__main__':
    collect_dir = "/raid/hzj/data/keypoint_project/Collection/20211025_rs"
    save_dir = os.path.join(collect_dir, "dataset.json")
    anns_dir = os.path.join(collect_dir, "xml")
    images_dir = os.path.join(collect_dir, "images")
    dic = get_xml_to_json_rs(anns_dir, images_dir)
    out_file = open(save_dir, "w")
    json.dump(dic, out_file, indent=4)
    out_file.close()


