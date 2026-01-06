# from src.algorithm.track import trackModule
# from .deepFaceRec import FaceRecModel
# from src.algorithm.match import matchModule
import os
import json
import cv2
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def Updatefacesbank(detModel, faceRecModel, jsonPath: str, bankPath: str):
    try:
        facebank_dir = os.path.join(bankPath, "facebank")
        if not os.path.exists(facebank_dir):
            os.mkdir(facebank_dir)
        with open(jsonPath, "r") as f:
            user_list = json.load(f)
        names = ["Unknown"]  # 用于存储所有姓名
        embeddings = []
        for user in user_list:
            embs = []
            name = user["person_name"]  # 获取用户的姓名
            photo = user["person_photo"]  # 获取用户的录入图片
            (photopath, _) = os.path.split(photo)  # 获取文件存储的路径
            frame = cv2.imread(photo)
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, det_result, facedet_result, _, _, _ = detModel.get_output(data)
            if len(facedet_result) == 1:
                embedding, img = faceRecModel.get_output(data, facedet_result)
            elif len(facedet_result) > 1:
                logger.info("{}检测到多张人脸!人脸数量{}!只有一张人脸会被保留。".format(name, len(facedet_result)))
                score = facedet_result[:, 4]
                index = np.argsort(-score)
                remain_det = facedet_result[index[0]: index[0]+1, :]
                embedding, img = faceRecModel.get_embedding(data, remain_det)
                # f_remain = frame.copy()
                for i in range(len(remain_det)):
                    bbox = facedet_result[i][0:4]
                    landmark = facedet_result[i][5:]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 10)
                    cv2.circle(frame, (landmark[0], landmark[1]), 10, (0, 0, 255), 4)
                    cv2.circle(frame, (landmark[2], landmark[3]), 10, (0, 255, 255), 4)
                    cv2.circle(frame, (landmark[4], landmark[5]), 10, (255, 0, 255), 4)
                    cv2.circle(frame, (landmark[6], landmark[7]), 10, (0, 255, 0), 4)
                    cv2.circle(frame, (landmark[8], landmark[9]), 10, (255, 0, 0), 4)
                cv2.imwrite(os.path.join(photopath, "remain_" + photo), frame)
            else:
                logger.info("{}不包含人脸，请检查。".format(name))
                continue
            for i in range(len(facedet_result)):
                bbox = facedet_result[i][0:4]
                landmark = facedet_result[i][5:]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 10)
                cv2.circle(frame, (landmark[0], landmark[1]), 10, (0, 0, 255), 4)
                cv2.circle(frame, (landmark[2], landmark[3]), 10, (0, 255, 255), 4)
                cv2.circle(frame, (landmark[4], landmark[5]), 10, (255, 0, 255), 4)
                cv2.circle(frame, (landmark[6], landmark[7]), 10, (0, 255, 0), 4)
                cv2.circle(frame, (landmark[8], landmark[9]), 10, (255, 0, 0), 4)
            cv2.imwrite(os.path.join(photopath, "detection_view_1.jpg"), frame)
            img[0].save(os.path.join(photopath, 'face_view_1.jpg'))
            user["person_detection_photo"] = os.path.join(photopath, "detection_view_1.jpg")
            user["person_face_photo"] = os.path.join(photopath, "face_view_1.jpg")
            embs.append(embedding)
            emb = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(emb)
            names.append(name)
        embeddings = torch.cat(embeddings)
        names = np.array(names)
        logger.info("更新 {} 张人脸!".format(len(names)))
        torch.save(embeddings, os.path.join(facebank_dir, 'facebank.pth'))
        np.save(os.path.join(facebank_dir, 'names'), names)
        # 更新完毕后将文件写入json
        with open(jsonPath, "w") as f:
            json.dump(user_list, f)
        logger.info("人脸更新完成!".format(len(names)))
        return True
    except Exception as e:
        logger.error(e)
        return False


def Updatefacebank(detModel, faceRecModel, user: dict):
    face_mask = [True, True, True, True, False,
                 True, True, True, True,
                 True, True, True, True,
                 True, True]
    try:
        # for user in user_list:
        embs = []
        name = user["person_name"]  # 获取用户的姓名
        photo = user["person_photo"]  # 获取用户的录入图片
        (photopath, _) = os.path.split(photo)  # 获取文件存储的路径
        frame = cv2.imread(photo)
        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, det_result, facedet_result, _, _, _ = detModel.get_output(data)
        if len(facedet_result) == 1:
            # print(facedet_result[:, face_mask])
            embedding, img = faceRecModel.get_face_embedding_image(data, facedet_result[:, face_mask])
        elif len(facedet_result) > 1:
            logger.info("{}检测到多张人脸!人脸数量{}!只有一张人脸会被保留。".format(name, len(facedet_result)))
            score = facedet_result[:, 4]
            index = np.argsort(-score)
            remain_det = facedet_result[index[0]: index[0]+1, :]
            embedding, img = faceRecModel.get_face_embedding_image(data, remain_det[:, face_mask])
            # f_remain = frame.copy()
            # for i in range(len(remain_det)):
            bbox = remain_det[0:4]
            landmark = remain_det[5:]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 10)
            cv2.circle(frame, (landmark[0], landmark[1]), 10, (0, 0, 255), 4)
            cv2.circle(frame, (landmark[2], landmark[3]), 10, (0, 255, 255), 4)
            cv2.circle(frame, (landmark[4], landmark[5]), 10, (255, 0, 255), 4)
            cv2.circle(frame, (landmark[6], landmark[7]), 10, (0, 255, 0), 4)
            cv2.circle(frame, (landmark[8], landmark[9]), 10, (255, 0, 0), 4)
            cv2.imwrite(os.path.join(photopath, "remain_" + photo), frame)
        else:
            logger.info("{}不包含人脸，请检查。".format(name))
            return False
        for i in range(len(facedet_result)):
            bbox = facedet_result[i][0:4]
            landmark = facedet_result[i][5:]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 10)
            cv2.circle(frame, (landmark[0], landmark[1]), 10, (0, 0, 255), 4)
            cv2.circle(frame, (landmark[2], landmark[3]), 10, (0, 255, 255), 4)
            cv2.circle(frame, (landmark[4], landmark[5]), 10, (255, 0, 255), 4)
            cv2.circle(frame, (landmark[6], landmark[7]), 10, (0, 255, 0), 4)
            cv2.circle(frame, (landmark[8], landmark[9]), 10, (255, 0, 0), 4)
        cv2.imwrite(os.path.join(photopath, "detection_view_1.jpg"), frame)
        cv2.imwrite(os.path.join(photopath, "face_view_1.jpg"), img[0])
        user["person_detection_photo"] = os.path.join(photopath, "detection_view_1.jpg")
        user["person_face_photo"] = os.path.join(photopath, "face_view_1.jpg")
        embs.append(embedding)
        emb = torch.cat(embs).mean(0, keepdim=True)
        return emb
        # embeddings.append(emb)
        # names.append(name)
        # embeddings = torch.cat(embeddings)
        # names = np.array(names)
        # logger.info("更新 {} 张人脸!".format(len(names)))
        # torch.save(embeddings, os.path.join(facebank_dir, 'facebank.pth'))
        # np.save(os.path.join(facebank_dir, 'names'), names)
        # # 更新完毕后将文件写入json
        # with open(jsonPath, "w") as f:
        #     json.dump(user_list, f)
        # logger.info("人脸更新完成!".format(len(names)))
        # return True
    except Exception as e:
        logger.error(e)
        return False


