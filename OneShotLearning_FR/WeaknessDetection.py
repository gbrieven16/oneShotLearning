from StyleEncoder import generate_image, get_encoding, move_and_show, DLATENT_DIR, GENERATED_IMAGES_DIR, DIRECTION_DIR
from Dataprocessing import FOLDER_DB, FOLDER_DIC, from_zip_to_data, TRANS, generate_synthetic_im
from Main import WITH_PROFILE, main_train

import os
import pickle
import tensorflow as tf
import numpy as np
import random
import torch.nn.functional as f

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
NB_CONSISTENT_W = 2
FOLDER_SYNTH_IM = "/data/gbrieven/train_synt_im/"
SIZE_SYNTH_TS = 200
SIZE_VALID = 100

NB_EPOCHS = 100
LR = 0.003


# =================================================
# Weakness Detector Training
# =================================================

""" ------------------- compose_pair ------------------------------------------
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
            face_dic.update(pickle.load(open(FOLDER_DB + FOLDER_DIC + "faceDic_" + db + ".pkl", "rb")))
        except (FileNotFoundError, EOFError) as e:
            print("The file " + FOLDER_DB + db + ".zip coundn't be found ... \n")
            fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + db + ".zip")
            face_dic = fileset.order_per_personName(TRANS, save=db)

    while len(pairs) < nb_input_pairs:
        pairs.append(compose_pair(face_dic, diff=DIFFERENT))

    for i, pair in enumerate(pairs):
        # -----------------------------------------------------------
        # 2. The second elements of the pairs are synthetized
        # -----------------------------------------------------------
        z2 = get_encoding(pair[1].db_path, pair[1].file_path)
        z2_prime = z2.copy()
        # direction = tf.Variable(get_initial_direction(dir_index))
        scale = 1.0  # std
        direction = tf.distributions.Normal(get_initial_direction(dir_name), scale)
        coef = tf.convert_to_tensor(COEF)
        z2_tf = tf.convert_to_tensor(z2)
        z2_prime[:8] = (z2_tf + coef * direction)[:8]  # Pq seulement 8 premiers éléments ???

        # Go back to an image so that the torch transform can be applied on
        synthetic_image = generate_image(z2_prime)  # PIL.Image
        pair[1].trans_image = TRANS(synthetic_image)

        # --------------------------------------------------------------
        # 3. Compute the distance between the embeddings of the pictures
        # in the pair
        # ---------------------------------------------------------------

        embed1 = pair[0].get_feature_repres(model)
        embed2 = pair[1].get_feature_repres(model)
        distance = f.pairwise_distance(embed1, embed2, 2)
        print("distance: " + str(distance))

        # --------------------------------------------------------------
        # 4. Apply some back propagation
        # ---------------------------------------------------------------
        # AIM = minimize the distance, i.e make it as close as possible to 0
        mse = tf.losses.mean_squared_error(0, distance)  # the loss function
        adam = tf.train.AdamOptimizer(learning_rate=LR)  # the optimizer
        if DIFFERENT:
            a = adam.minimize(mse, var_list=direction)  # this runs one step of gradient descent
        else:
            a = adam.maximize(mse, var_list=direction)
        return direction, a


def run(w, optim):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NB_EPOCHS):
            sess.run(optim)  # run for the number of training steps
        w = sess.run(w)

        print(w)  # this will output our current weights after training
        return w


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    test_id = 0
    dir_name = "smile"
    w = None
    model = "models/dsgbrieven_filteredcfp_humFilteredlfw_filteredfaceScrub_humanFiltered_3245_1default_" \
            "70_triplet_loss_nonpretrained.pt"

    db_source_list = ["cfp_humFiltered"] #, "lfw_filtered", "gbrieven_filtered", "faceScrub_humanFiltered"]

    if test_id == 0:
        # -------------------------------------
        # Find consistent initial directions
        # -------------------------------------
        w = np.random.rand(18, 512)
        dir_name = str(NB_CONSISTENT_W + 1)

    if test_id == 1:
        # -------------------------
        # Train weakness detector
        # -------------------------
        weights, optimizer = main_wd(db_source_list, model, dir_name=dir_name)
        w = run(weights, optimizer)
        w_id = dir_name + "_" + "1"

    # -------- Store the latent direction ----------
    fname = DIRECTION_DIR + dir_name + ".npy"
    np.save(fname, w.eval())
    print("The latent direction after training has been saved as " + fname)

    if test_id in [0, 1]:
        # -----------------------------------------
        # Check consistency of the synthetic image
        # -----------------------------------------

        # Take some latent representation
        nb_test = 10
        dlatent_name_list = os.listdir(DLATENT_DIR)

        for i in range(nb_test):
            latent_vector = np.load(DLATENT_DIR + dlatent_name_list[i])
            save_result = GENERATED_IMAGES_DIR + dir_name + "/" + dlatent_name_list[i].split(".")[0] + ".jpg"
            move_and_show(latent_vector, w, COEF, save_result)

    if test_id == 2:
        # ------------------------------------------------
        # Generate synthetic images from learnt direction
        # ------------------------------------------------
        db = FOLDER_DB + "cfp_humFiltered.zip"
        train_fset = generate_synthetic_im(db, nb_additional_images=2, directions=[dir_name])
        val_fileset = from_zip_to_data(False, fname=FOLDER_DB + db_source_list[0] + ".zip", max_nb_entry=SIZE_VALID)
        sets_list = [train_fset, val_fileset]

        # ------------------------------------------------
        # Retrain Model On
        # ------------------------------------------------
        fname = []
        for i, db_source in enumerate(db_source_list):
            fname.append(FOLDER_DB + db_source + "zip")

        main_train(sets_list, fname, name_model=model, with_synt=True)