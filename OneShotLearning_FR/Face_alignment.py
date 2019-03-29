# Source: https://github.com/Puzer/stylegan-encoder

import numpy as np
import scipy.ndimage
import os
import PIL.Image
import sys
import bz2
import dlib
from keras.utils import get_file
from io import BytesIO
import zipfile

#########################################
#       GLOBAL VARIABLES                #
#########################################

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
RAW_IMAGES_DIR = "/data/gbrieven/FFHQ500"  # "/data/gbrieven/FFHQ500"

ALIGNED_IMAGES_DIR = "/data/gbrieven/FFHQ500_aligned/"
ZIP_FILE = None  # "/data/gbrieven/FFHQ500.zip"


#########################################
#    CLASS LandmarksDetector            #
#########################################

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector()  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image_path=None, image_obj=None, loaded=None):

        if loaded is not None:
            img = loaded
        elif image_path is not None:
            img = dlib.load_rgb_image(image_path)
        else:
            img = np.array(image_obj)

        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks


#########################################
#       "GLOBAL OBJECT"                 #
#########################################

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

print("LandmarksDetector Definition ... \n")
landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)

#########################################
#       FUNCTIONS                       #
#########################################

"""
IN: img: PIL.Image.Image object 
    loaded: 
"""


def align_faces(img, save_name=None, loaded=None, output_size=(256, 256), transf_size=(512, 512), enable_padding=True):

    # ------------- Go Though all the detected faces (here only one) -----------------
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(image_obj=img, loaded=loaded), start=1):
        lm = np.array(face_landmarks)
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # -----------------------------
        # Calculate auxiliary vectors.
        # -----------------------------
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # -------------------------------
        # Choose oriented crop rectangle.
        # -------------------------------
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # -----------------------------
        # Shrink.
        # -----------------------------
        shrink = int(np.floor(qsize / output_size[1] * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # -----------------------------
        # Crop.
        # -----------------------------
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # -----------------------------
        # Pad.
        # -----------------------------
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # -----------------------------
        # Transform.
        # -----------------------------
        img = img.transform(transf_size, PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if max(output_size) < max(transf_size):
            img = img.resize(output_size, PIL.Image.ANTIALIAS)

        # -----------------------------
        # Save aligned image.
        # -----------------------------
        if save_name is not None:
            img.save(ALIGNED_IMAGES_DIR + save_name, 'jpeg')
        return img


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":
    pass