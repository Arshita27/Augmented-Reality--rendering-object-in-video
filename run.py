import math
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from augment import AR
from config import ConfigNode


def run(cfg, ar, ref_img, frame):

    h, w = frame.shape[0], frame.shape[1]
    resized_ref_img = cv2.resize(ref_img, (int(w), int(h)), interpolation = cv2.INTER_AREA)

    raw_image_list = [resized_ref_img, frame]

    gray_image_list = ar.convert_2_gray(raw_image_list)

    kps_list, descs_list = ar.get_feature_desc(gray_image_list)

    ar.draw_keypoints(raw_image_list, gray_image_list, kps_list, descs_list, )

    matches = ar.get_best_matches(gray_image_list[0:2], kps_list[0:2], descs_list[0:2], )

    if len(matches) >= 4:
        H = ar.get_homography_matrix(matches, kps_list[0:2])

        ht, wd  = raw_image_list[0].shape[0], raw_image_list[0].shape[1]

        pts = np.float32([[0, 0],  [0, ht],  [wd , ht ], [wd , 0], ]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, H)

        frame = cv2.polylines(raw_image_list[1], [np.int32(dst)], True, 255, 2, cv2.LINE_AA)

        # cv2.imwrite(os.path.join(cfg.DATASET.OUTPUT_DIR, 'frame.jpg'), frame)

        # ####################################

        projection = ar.projection_matrix(H)

        frame = ar.render(frame, projection, raw_image_list[0])

    else:
        print("Insufficient matches found!")

    return frame

import time
if __name__== "__main__":

    cfg_file = "config.yml"

    with open(cfg_file, "r") as ymlfile:
        node = yaml.load(ymlfile)

    cfg = ConfigNode(node)

    ar = AR(cfg)

    ref_img = cv2.imread(os.path.join(cfg.DATASET.INPUT_DIR, cfg.DATASET.REF_IMG))

    cap = cv2.VideoCapture(cfg.DATASET.VIDEO_PATH)

    count = 0
    while(cap.isOpened()):
        start_time = time.time()
        ret, frame = cap.read()

        cv2.imwrite('frame.png', frame)
        frame2 = run(cfg, ar, ref_img, frame)

        count +=1
        cv2.imshow('frame',frame2)
        # print(time.time() - start_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    print(count)
