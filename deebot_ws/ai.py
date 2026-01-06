import requests
import json
from PIL import Image
import io
import base64
import cv2
import numpy as np
from deebot_ros import Deebot,DeebotControlParam
import rospy

import time
from datetime import datetime


class AIPlan:
    def getplan(self, img)-> np.ndarray:
        res = np.array([[1.,2.],[1.,2.]]) # [[x,y],[x,y]]
        assert res.shape[1] == 2
        return res 


def img_convert(img_cv):
    img = cv2.resize(img_cv, (1080, 1920)).astype(np.float16)

    img *= 0.7
    img = img.astype(np.uint8)

    img = Image.fromarray(img)

    img_byte_array = io.BytesIO()
    img.save(img_byte_array,format='JPEG')
    img_byte_array = img_byte_array.getvalue()

    img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

    return img_base64


if __name__ == '__main__': 
    deebot = Deebot(sim=False,keyboard_control=False,init_node=True)
    param = DeebotControlParam( l_k=0.05,                # 目标点距离速度增益
                                lfc=0.4,                 # 目标点基础距离
                                kp_v=0.5,                # 速度比例增益
                                kp_s=1.5,                # 距离比例增益
                                max_speed=0.3,           # [m/s] 最大速度
                                wheel_l=0.23,            # [m] 左右轮之间的距离
                                stop_s=0.01,             # [m] 到达终点时的停止距离
                                stop_dyaw=5,             # [deg] 旋转停止角度
                                back_dyaw=100,           # [deg] 后退角度，当机器人与目标点连线夹角超过此值时将后退
                                k_rot_scale=0.8,         # w = k_rot_scale * dyaw / 0.5 PI
                                rot_back_dyaw=80,        # [deg] 如果规划角度（最近路径点）与机器人的角度夹角大于此值，则进行旋转
                                max_over_s=0.2,          # [m] 最大额外行驶距离，即计划行驶距离加上此值，若超过则认为已到达并进行旋转
                                min_v=0.1,               # [m/s] 最小线速度
                                min_w=0.2)               # [rad/s] 最小角速度
    deebot.set_param(param)

    # ai = AIPlan()
    #rate = rospy.Rate(rospy.Duration(0.5))

    url = "http://123.60.219.36:6006"

    while not rospy.is_shutdown():
        img_cv, ts = deebot.get_img()
        bump_states:list = deebot.get_bump_states()

        # key 如下
        # uint8 RANGE_TYPE_FRONT_BUFFER = 0
        # uint8 RANGE_TYPE_SIDE_IN = 1
        # uint8 RANGE_TYPE_DOWN_IN = 2
        # uint8 RANGE_TYPE_ULTRASOUND = 3
        edge_states:dict = deebot.get_edge_states()
        if img_cv is None or len(edge_states)==0 or bump_states[0] is None:
            print(f"img_cv is None: {img_cv is None}\nedge_states is empty:{len(edge_states)==0}\nbump_states is None :{bump_states[0] is None}")
            continue

        print(f" get img {img_cv.shape}, bump_states:{bump_states}, edge_states:{edge_states}")
        #continue

        # print(type())
        img_base64 = img_convert(img_cv)

        data={
            'image': img_base64,
            'name': str(ts),
            'ts0': str(datetime.fromtimestamp(time.time()))
        }
        headers = {
            'Content-Type':'application/json'
        }

        response = requests.post(url,json=data,headers=headers)
        print(response)
        response_data = response.json()
        print("Response data",response_data)

        path = np.array(response_data['data']['trajectory']).astype(np.float16)
  

        # path = np.array([
        #     [0.1, 0],
        #     [0.2, 0],
        #     [0.4, 0.1]
        # ])


        if len(path):
            path -= 360
            path /= 1000.
            print(path)
            deebot.pub_path(path,ts)
        else:
            print("no path")

        time.sleep(0.001)
        # time.sleep(1)
    
    cv2.destroyAllWindows()


