
def conform_init_box(frame, det_result):
    image_w = frame.shape[1]
    image_h = frame.shape[0]

    center_dist = []
    conform_init_index = -1

    for box_index, result in enumerate(det_result):
        # bbox = result[0: 4]  # x1 y1 x2 y2
        bbox = result["body_box"]
        # 归一化0-1范围计算距离中心点距离
        box_center_w = ((bbox[0] + bbox[2]) / 2) / image_w
        box_center_h = ((bbox[1] + bbox[3]) / 2) / image_h

        box_center_dist = abs(0.5 - box_center_h) + abs(0.5 - box_center_w)
        center_dist.append(box_center_dist)

    # 去最小值索引

    min_dist_index = center_dist.index(min(center_dist))

    # best_init_box = det_result[min_dist_index][0:4]
    best_init_box = det_result[min_dist_index]["body_box"]
    # 选择初始化ID特征，选择中间位置，不选择靠近边上位置

    center_x = (best_init_box[0] + best_init_box[2]) / 2.0
    ratio_w = center_x / image_w

    center_y = (best_init_box[1] + best_init_box[3]) / 2.0
    ratio_h = center_y / image_h

    # 判断目标处于中间位置，作为初始化目标并提取特征
    if ratio_w > 0.2 and ratio_w < 0.8 and ratio_h > 0.01 and ratio_h < 0.99:
        conform_init_index = min_dist_index

    return conform_init_index
