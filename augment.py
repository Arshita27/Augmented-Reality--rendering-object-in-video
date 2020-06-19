import math
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from obj_loader import *


class AR():
    def __init__(self, cfg ):
        self.cfg = cfg
        self.root_path = cfg.DATASET.INPUT_DIR
        self.result_dir = cfg.DATASET.OUTPUT_DIR

    def read_image(self, ):
        '''
        Reads raw image.
        '''

        return [cv2.imread(os.path.join(self.root_path, file_path))
                for file_path in self.cfg.DATASET.INPUT_IMG_LIST]

    def convert_2_gray(self, img_list: List):
        '''
        Converts image into gray scale.
        '''

        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]

    def get_feature_desc(self, img_list: List):
        '''
        Feature descriptor
        '''

        kps_list=[]
        descs_list=[]
        for img in img_list:
            if self.cfg.FEATURES.FEATURE_DESCRIPTORS == "SIFT":
                get_features = cv2.xfeatures2d.SIFT_create()

            elif self.cfg.FEATURES.FEATURE_DESCRIPTORS == "SURF":
                get_features = cv2.xfeatures2d.SURF_create()
                #NOTE: add threshold

            elif self.cfg.FEATURES.FEATURE_DESCRIPTORS == "ORB":
                get_features = cv2.ORB_create()

            else:
                raise ValueError("'{}' feature descriptor is not defined. Check Config.yaml file. \
                ".format( self.cfg.FEATURES.FEATURE_DESCRIPTORS))

            (kps, descs) = get_features.detectAndCompute(img, None)

            kps_list.append(kps)
            descs_list.append(descs)

        return kps_list, descs_list

    def draw_keypoints(self, raw_img_list: List, gray_img_list: List, kps_list, descs_list):
        '''
        Draw key points
        '''

        for i, raw_img in enumerate(raw_img_list):
            raw_img_copy = raw_img.copy()
            img_keypoints = cv2.drawKeypoints(
                                    gray_img_list[i],
                                    kps_list[i],
                                    raw_img_copy,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(self.result_dir, "img_"+str(i)+"_keyfeatures.png"),
                        img_keypoints)
        print(f'Saving images with keypoints.')


    def get_best_matches(self, img, kps, descs):
        '''
        Feature Matching
        '''

        fm = tuple(k.strip() for k in self.cfg.FEATURES.FEATURE_MATCHING[1:-1].split(','))

        if fm[0] == "Brute_Force" and fm[1] == "NORM_L2":
            assert (self.cfg.FEATURES.FEATURE_DESCRIPTORS == "SIFT" or
                    self.cfg.FEATURES.FEATURE_DESCRIPTORS == "SURF" ), "Check FEATURE_DESCRIPTORS"
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descs[0], descs[1], k=2)

        if fm[0] == "Brute_Force" and fm[1] == "NORM_HAMMING":
            assert (self.cfg.FEATURES.FEATURE_DESCRIPTORS == "ORB" or
                    self.cfg.FEATURES.FEATURE_DESCRIPTORS == "BRIEF" or
                    self.cfg.FEATURES.FEATURE_DESCRIPTORS == "BRISK"), "Check FEATURE_DESCRIPTORS"
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(descs[0], descs[1], k=2)

        good = []
        for m,n in matches:
            if m.distance < self.cfg.FEATURES.FEATURE_MATCHING_THRESHOLD*n.distance:
                good.append([m])

        res_img = cv2.drawMatchesKnn(img[0], kps[0], img[1], kps[1], good, None, flags=2)
        cv2.imwrite(os.path.join(self.result_dir, 'matched_points.jpg'), res_img)

        return np.asarray(good)


    def get_homography_matrix(self, matches, kps):
        '''
        get homography matrix
        '''

        if len(matches[:,0]) >= 4:
            src = np.float32([ kps[0][m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ kps[1][m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        return H

    def projection_matrix(self, homography):
        """
        From the camera calibration matrix and the estimated homography
        compute the 3D projection matrix

        NOTE: taken from https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
        """

        camera_parameters = np.array(self.cfg.CAMERA_PARAMETERS)
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(camera_parameters, projection)

    def render(self, img, projection, model, color=False):
        """
        Render a loaded obj model into the current video frame

        NOTE: taken from https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/

        """
        scale_factor = self.cfg.RENDERING.SCALE_FACTOR
        scale_matrix = np.eye(3) * scale_factor
        h, w = model.shape[0], model.shape[1]

        obj = OBJ(self.cfg.DATASET.RENDERED_OBJ, swapyz=True)
        vertices = obj.vertices

        for face in obj.faces:
            face_vertices = face[0] # vertices that make the face.
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)

            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            else:
                # if face[-1]:
                __import__("pdb").set_trace()
                color = hex_to_rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(img, imgpts, color)

        return img
