from StyleEncoder import generate_image, get_encoding, move_and_show, DLATENT_DIR, GENERATED_IMAGES_DIR, DIRECTION_DIR
from Dataprocessing import FOLDER_DB, FOLDER_DIC, from_zip_to_data, TRANS, generate_synthetic_im
from Main import WITH_PROFILE, main_train, load_model

# To transform model
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnx
from onnx_tf.backend import prepare

import torch
import torch.nn.functional as f

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
#                    GLOBAL VARIABLES
# ================================================================


SEED = 8
TRAINING_SIZE = 500
DIFFERENT = True
COEF = 1
NB_CONSISTENT_W = len(os.listdir(DIRECTION_DIR))
SIZE_SYNTH_TS = 200
SIZE_VALID = 100

NB_EPOCHS = 100
LR = 0.003

try:
    MEAN = sys.argv[1]
    STD = sys.argv[2]
    PROP_NO_VARIATION = sys.argv[3]
except IndexError:
    MEAN = 0
    STD = 0.005
    PROP_NO_VARIATION = 0.4

W_DIM1 = 18
W_DIM2 = 512

# =================================================
# Weakness Detector Training
# =================================================

""" ------------------- compose_pair ----------------------------------------
This function returns a pair of pictures
IN: face_dic: dictionary where the key is the name of the person and the 
value is a list of pictures of this person 
    diff: the pair is composed of pictures of the same person if False 
--------------------------------------------------------------------------- """


def compose_pair(face_dic, diff=True):
    people = list(face_dic.keys())
    # ------------- Picture 1 -----------------
    pers1 = random.choice(people)
    people.remove(pers1)
    pict1 = face_dic[pers1].pop()
    if len(face_dic[pers1]) < 2:
        del face_dic[pers1]

    # ------------- Picture 2 -----------------
    pers2 = random.choice(people) if diff else pers1

    pict2 = face_dic[pers2].pop()
    if len(face_dic[pers2]) < 2:
        del face_dic[pers2]

    return pict1, pict2


def get_initial_direction(dir_name):
    return np.load(DIRECTION_DIR + dir_name + ".npy")


""" ------------------ main_wd ------------------------------------------------
This function defines the loss and the optimizer of the weakness detectors
----------------------------------------------------------------------------- """


def main_wd(db_source_list, model, nb_input_pairs=TRAINING_SIZE, dir_name="smile"):
    # -----------------------------------------------------------
    # 1. Build Input
    # Input = list of tuples where the 2 elements are FaceImage
    # -----------------------------------------------------------

    face_dic = {}
    pairs = []

    for i, db in enumerate(db_source_list):
        try:
            face_dic.update(pickle.load(open(FOLDER_DIC + "faceDic_" + db + ".pkl", "rb")))
        except (FileNotFoundError, EOFError) as e:
            print("The file " + FOLDER_DB + db + ".zip coundn't be found ... \n")
            fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + db + ".zip")
            face_dic = fileset.order_per_personName(TRANS, save=db)

    # -------- Put constraints on the probability distribution of the variable w to optimize -----------------
    scale = np.full((W_DIM1, W_DIM2), 0.01)
    direction = tf.distributions.Normal(get_initial_direction(dir_name), scale)
    while len(pairs) < nb_input_pairs:
        pairs.append(compose_pair(face_dic, diff=DIFFERENT))
        direction = tf.convert_to_tensor(get_initial_direction(dir_name).astype(np.float32))

    for i, pair in enumerate(pairs):
        # -----------------------------------------------------------
        # 2. The second elements of the pairs are synthetized
        # -----------------------------------------------------------
        z2 = get_encoding(pair[1].db_path, pair[1].file_path)

        # Go back to an image so that the torch transform can be applied on
        synthetic_image = generate_image(z2 + direction.eval())  # PIL.Image
        pair[1].trans_image = TRANS(synthetic_image)

        # --------------------------------------------------------------
        # 3. Compute the distance between the embeddings of the pictures
        # in the pair
        # ---------------------------------------------------------------

        embed1 = pair[0].get_feature_repres(model)
        embed2 = pair[1].get_feature_repres(model)
        distance = f.pairwise_distance(embed1, embed2, 2).data[0].detach().cpu().numpy()
        print("distance: " + str(distance))

        # --------------------------------------------------------------
        # 4. Apply some back propagation
        # ---------------------------------------------------------------
        # AIM = minimize the distance, i.e make it as close as possible to 0
        mse = tf.losses.mean_squared_error(0, distance)  # the loss function
        adam = tf.train.AdamOptimizer(learning_rate=LR)  # the optimizer
        if DIFFERENT:
            a = adam.minimize(mse, var_list=[direction])  # this runs one step of gradient descent
        else:
            a = adam.maximize(mse, var_list=[direction])
        return direction, a


def run(w, optim):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NB_EPOCHS):
            sess.run(optim)  # run for the number of training steps
        w = sess.run(w)

        print(w)  # this will output our current weights after training
        return w


"""
This function randomly sets prop_O of the element of x to 0 
x: np array of dimensions (18, 512)
"""


def set_to_zero(x, prop_O=PROP_NO_VARIATION):
    for i in range(W_DIM1):
        # Pick randomly indices for the first dimension
        indices_2 = np.random.choice(np.arange(W_DIM2), replace=False, size=int(prop_O * W_DIM2))
        x[i][indices_2] = 0
    return x


"""
This function converts a pytorch model to a tensorflow one 
IN: pytorch model (type nn.Module)
"""


def model_convert(model):
    # ------------------- From Pytorch to ONNX form ------------------------
    input_shape = (32, 3, 200, 150)
    model_onnx_path = "torch_model.onnx"

    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    print("Export of torch_model.onnx complete!")

    # ------------------- From ONNX to Tensorflow form ---------------------
    onnx_model = onnx.load(model_onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model)
    return tf_rep


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":
    test_id = 0
    set_to_zero(np.random.normal(MEAN, STD, (W_DIM1, W_DIM2)))
    dir_name_list = ["smile"]
    w_list = []
    nb_directions = 3
    model_name = "models/dsgbrieven_filteredlfw_filtered_8104_1default_70_cross_entropy_pretautoencoder.pt"

    db_source_list = ["cfp_humFiltered"]  # lfw_filtered"]  # , , "gbrieven_filtered", "faceScrub_humanFiltered"]

    if test_id == 0:
        print("-----------------------------------------")
        print("TEST0: Find consistent initial directions")
        print("-----------------------------------------\n")
        # w = np.random.rand(18, 512)
        dir_name_list = []
        for i in range(nb_directions):
            w_list.append(set_to_zero(np.random.normal(MEAN, STD, (W_DIM1, W_DIM2))))
            dir_name_list.append(str(NB_CONSISTENT_W + i + 1))
            fname = DIRECTION_DIR + dir_name_list[-1] + ".npy"
            np.save(fname, w_list[-1])

            print("The latent direction after training has been saved as " + fname)

    if test_id == 1:
        print("------------------------------")
        print("TEST1: Train weakness detector")
        print("------------------------------\n")
        model = load_model(model_name)
        weights, optimizer = main_wd(db_source_list, model, dir_name=dir_name_list[-1])
        w_list = [run(weights, optimizer)]
        w_id = dir_name_list[-1] + "_" + "1"

        # -------- Store the latent direction ----------
        fname = DIRECTION_DIR + dir_name_list[-1] + "afterTrained.npy"
        np.save(fname, w_list[-1].eval())
        print("The latent direction after training has been saved as " + fname)

    if test_id in [0, 1]:
        print("-----------------------------------------")
        print("Check consistency of the synthetic images")
        print("-----------------------------------------")

        # Take some latent representation
        nb_test = 5
        dlatent_name_list = os.listdir(DLATENT_DIR)[10:]

        for j, w in enumerate(w_list):
            for i in range(nb_test):
                latent_vector = np.load(DLATENT_DIR + dlatent_name_list[i])
                save_result = GENERATED_IMAGES_DIR + dir_name_list[j] + "_" + dlatent_name_list[i].split(".")[
                    0] + str(MEAN) + "_" + str(STD) + "_" + str(PROP_NO_VARIATION) + ".jpg"
                move_and_show(latent_vector, w, COEF, save_result)

    if test_id == 2:
        print("-----------------------------------------------")
        print("Generate synthetic images from learnt direction")
        print("-----------------------------------------------")
        model = load_model(model_name)

        db = FOLDER_DB + "cfp_humFiltered.zip"
        train_fset = generate_synthetic_im(db, nb_additional_images=2, directions=[dir_name_list[-1]])
        val_fileset = from_zip_to_data(False, fname=FOLDER_DB + db_source_list[0] + ".zip", max_nb_entry=SIZE_VALID)
        sets_list = [train_fset, val_fileset]

        # ------------------------------------------------
        # Retrain Model On
        # ------------------------------------------------
        fname = []
        for i, db_source in enumerate(db_source_list):
            fname.append(FOLDER_DB + db_source + "zip")

        main_train(sets_list, fname, name_model=model, with_synt=True)
