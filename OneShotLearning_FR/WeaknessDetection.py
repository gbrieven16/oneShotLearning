from StyleEncoder import generate_image, get_encoding
from Dataprocessing import FOLDER_DB, FOLDER_DIC, from_zip_to_data, TRANS
from Main import WITH_PROFILE

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

NB_EPOCHS = 100
LR = 0.003

# =================================================
# Weakness Detector Training
# =================================================

"""
This function returns a pair of pictures
IN: face_dic: dictionary where the key is the name of the person and the 
value is a list of pictures of this person 
    diff: the pair is composed of pictures of the same person if False 
"""
def compose_pair(face_dic, diff=True):
    people = list(face_dic.keys())
    # ------------- Picture 1 -----------------
    pers1=random.choice(people)
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


def get_initial_direction():
    return np.load('ffhq_dataset/latent_directions/smile.npy')


""" ------------------ main_wd ------------------------------------------------
This function defines the loss and the optimizer of the weakness detectors
----------------------------------------------------------------------------- """

def main_wd(db_source_list, model, nb_input_pairs=TRAINING_SIZE):

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
        direction = tf.Variable(get_initial_direction())
        coef = tf.convert_to_tensor(COEF)
        z2_tf = tf.convert_to_tensor(z2)
        z2_prime[:8] = (z2_tf + coef * direction)[:8]  # Pq seulement 8 premiers éléments ???

        # Go back to an image so that the torch transform can be applied on
        synthetic_image = generate_image(z2_prime) # PIL.Image
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
        a = adam.minimize(mse, var_list=direction) # this runs one step of gradient descent
        return direction, a


def run(weights, optimizer):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NB_EPOCHS):
            sess.run(optimizer)  # run for the number of training steps
        weights = sess.run(weights)

        print(weights)  # this will output our current weights after training

# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    db_source_list = ["cfp_humFiltered", "lfw_filtered", "gbrieven_filtered", "faceScrub_humanFiltered"]
    model = "models/dsgbrieven_filteredcfp_humFilteredlfw_filteredfaceScrub_humanFiltered_3245_1default_" \
            "70_triplet_loss_nonpretrained.pt"

    weights, optimizer = main_wd(db_source_list, model)
    run(weights, optimizer)
