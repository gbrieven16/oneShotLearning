import os
import matplotlib
matplotlib.use('Agg')  # TkAgg

import torch
import random
import time
import pickle
from random import shuffle, Random
import numpy as np

from Dataprocessing import from_zip_to_data, extract_randomly_elem, TRANS, FOLDER_DB, FaceImage, \
    DIST_METRIC, FOLDER_DIC, DLATENT_DIR, SEPARATOR
from Visualization import multi_line_graph, fr_in_csv
from Main import WITH_PROFILE, load_model

# =====================================================================================================================
#                                   GLOBAL VARIABLES
# WITH_SYNTHETIC_DATA: if True, the synthetic images are used as instances of the probe to enforce the comparison
#                      (BUT not in the gallery)
# =====================================================================================================================

NB_PROBES = 35
NB_INST_PROBES = 1

WITH_VOTE = True
SIZE_GALLERY = 20  # Nb of people to consider
NB_IM_PER_PERS = 8
TOLERANCE = 5  # 3 Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person
NB_REPET = 7  # Nb of times test is repeated (over a different set of probes)
THRESHOLDS_LIST = list(np.arange(0, 0.05, 0.0005))  # For MeanSquare
WITH_LATENT_REPRES = False
DETAILED_PRINT = False
TRESH_DIST_SEP_PRED_PERC = 0.5
MAX_NB_PERS = 3  # Maximum nb of predicted people
N = [1, 3, 5, 10, 20, 30]

WITH_SYNTHETIC_DATA = False
if WITH_SYNTHETIC_DATA or WITH_LATENT_REPRES:
    NB_IM_PER_PERS = None  # No restriction, otherwise you may not get the synth images in face_dic


# ======================================================================
#                    CLASS: Probe
# ======================================================================

class Probe:
    def __init__(self, person, pictures, index_pict):
        self.person = person
        self.pictures = pictures  # list of pictures (basic + synthetic / basic + extra)
        self.index = index_pict  # list of indice(s)

        self.dist_pers = {}  # dic where the key is the person's name and the value is the list of distances from probe
        self.dist_avg_pers = {}
        self.vote_pers = {}

        self.fn_list = [0] * len(THRESHOLDS_LIST)  # Number of false negative (i.e. incorrect rejection)
        self.fp_list = [0] * len(THRESHOLDS_LIST)  # Number of false positive (i.e incorrect acceptance)

    """ ============================= 
        avg_dist 
    ================================= """

    def avg_dist(self, person, nb_pictures):
        if 1 < nb_pictures:
            self.dist_avg_pers[person] = sum(self.dist_pers[person]) / nb_pictures

    """ ============================= 
        median_dist 
    ================================= """

    def median_dist(self, person, nb_pictures):
        if 1 < nb_pictures:
            try:  # TOCHECK: error occurred with 'lfwAllison_Janney'
                self.dist_avg_pers[person] = np.median(self.dist_pers[person])
                # print("median: " + str(self.dist_avg_pers[person]))
            except KeyError:
                print("KEYERR in median dist: " + person + " not found as key in " + str(self.dist_pers) + "\n")

    """ ============================= 
        predict_from_dist 
    ================================= """

    def predict_from_dist(self, res_acc_dist, top_n_list):
        ordered_people = sorted(self.dist_avg_pers.items(), key=lambda x: x[1])

        # ------------- Build List of predicted people ---------------------
        avg_dist_sep = get_avg_sep(ordered_people)
        i = 1
        pred_pers_dist = [ordered_people[0][0]]

        # -------------Store all the predicted people ---------------------
        dist_diff = ordered_people[i][1] - ordered_people[0][1]

        while i < len(ordered_people) - 1 and dist_diff < avg_dist_sep - TRESH_DIST_SEP_PRED_PERC * avg_dist_sep \
                and len(pred_pers_dist) < MAX_NB_PERS:
            pred_pers_dist.append(ordered_people[i][0])
            dist_diff = ordered_people[i + 1][1] - ordered_people[i][1]
            i += 1

        # ------------- Check correctness ---------------------
        if self.person in pred_pers_dist:
            res_acc_dist["nb_correct_dist"] += 1 / len(pred_pers_dist)
            # print("DELETE: Correct and dist diff is " + str(dist_diff))
        else:
            res_acc_dist["nb_mistakes_dist"] += 1
            # print("DELETE: Not correct and dist diff is " + str(dist_diff))

        for j, n in enumerate(N):
            people = [person for _, (person, dist) in enumerate(ordered_people[:n])]
            top_n_list[j] = top_n_list[j] + 1 if self.person in people else top_n_list[j]

    """ ============================= 
        pred_from_vote 
    ================================= """

    def pred_from_vote(self, DETAILED_PRINT, res_vote):

        if 0 < len(self.vote_pers):
            sorted_vote = sorted(self.vote_pers.items(), key=lambda x: x[1], reverse=True)
            if DETAILED_PRINT:
                print("\n ------------- The current probe is: " + str(self.person) + " ----------------------- ")
                print("Voting system is represented by: " + str(sorted_vote) + "\n")

            # Check if there's any equality in the votes
            nb_equal_vote = -1
            for i, (person, score) in enumerate(sorted_vote):
                if score == sorted_vote[0][1] and nb_equal_vote < MAX_NB_PERS:
                    nb_equal_vote += 1
                else:
                    break

            pred_person = [pers for i, (pers, score) in enumerate(sorted_vote) if i < nb_equal_vote + 1]

            if self.person in pred_person:
                res_vote["nb_correct"] += 1 / (nb_equal_vote + 1)
                if DETAILED_PRINT: print("The correct predicted person is: " + str(pred_person) + "\n")
            else:
                res_vote["nb_mistakes"] += 1
        else:
            if DETAILED_PRINT: print("The person wasn't recognized as belonging to the gallery!\n")
            res_vote["nb_not_recognized"] += 1

    """ ============================= 
        compute_false 
    ================================= """

    def compute_false(self):
        for i, thresh in enumerate(THRESHOLDS_LIST):
            for person, distances in self.dist_pers.items():
                for j, dist in enumerate(distances):
                    # ---- CASE 1: FF (incorrect prediction ; pred = different) --------
                    # print("Thresh is " + str(thresh) + " and dist is " + str(dist))
                    if person == self.person and thresh < dist:
                        self.fn_list[i] += 1
                    # ---- CASE 2: FP (incorrect prediction ; pred = similar) --------
                    elif person != self.person and dist < thresh:
                        self.fp_list[i] += 1


# ======================================================================
#                    CLASS: FaceRecognition
# ======================================================================


class FaceRecognition:
    def __init__(self, model_path, db_source=None):
        self.acc_model = [0, 0]  # First element: correct/not (1/0) ; Second element: Counter
        self.pos_recall = [0, 0]  # First element: correct/not (1/0) ; Second element: Counter of positives
        self.neg_recall = [0, 0]  # First element: correct/not (1/0) ; Second element: Counter of positives

        self.probes = []  # list of NB_REPET lists of lists (person, picture, index_pict)

        db_source = ["testdb"] if db_source is None else db_source
        self.k_considered = []
        self.distances = {}

        # -------- Model Loading ----------------
        if model_path is not None:
            model = load_model(model_path)
            try:
                self.siamese_model = model.network.cuda() if torch.cuda.is_available() else model.network
            except AttributeError:
                self.siamese_model = model.cuda() if torch.cuda.is_available() else model

        # ------- Gallery Definition --------------
        self.gallery = get_gallery(SIZE_GALLERY, db_source)
        self.nb_sim = max(len(self.gallery[next(iter(self.gallery))]) - 1, 1)  # nb of pictures "similar" to a probe
        self.nb_dif = self.nb_sim * (len(self.gallery) - 1)  # nb of pictures "different" from a probe

        people_gallery = list(self.gallery.keys())

        # ------- Build NB_REPET probes --------------
        for rep_index in range(NB_REPET):

            # Pick different people (s.t. the test pictures are related to different people)
            shuffle(people_gallery)

            probes_k = []
            for pers_i, person in enumerate(people_gallery[:NB_PROBES]):

                # ---------- Pick NB_INST_PROBES random picture(s) ------------
                if not WITH_SYNTHETIC_DATA:
                    probe_pict, indexes_probe = extract_randomly_elem(NB_INST_PROBES, self.gallery[person])

                # ---------- Pick one picture having stored synthetised version ------------
                else:
                    try:
                        probe_pict, indexes_probe = get_synth_pict(self.gallery[person])
                    except TypeError: # No synthetic data was found
                        return

                probes_k.append(Probe(person, probe_pict, indexes_probe))

            self.probes.append(probes_k)

        remove_synth_data(self.gallery)

    '''---------------- recognition ---------------------------
     This function identifies the person on each test picture
     ----------------------------------------------------------'''

    def recognition(self, index):

        self.k_considered.append(index)

        res_vote = {"nb_not_recognized": 0, "nb_mistakes": 0, "nb_correct": 0}
        res_dist = {"nb_mistakes_dist": 0, "nb_correct_dist": 0}
        res_dist_topn = [0 for _ in range(len(N))]

        self.acc_model = [0, 0]
        self.pos_recall = [0, 0]
        self.neg_recall = [0, 0]

        # -----------------------
        # Go through each probe
        # ----------------------- #
        for i, probe in enumerate(self.probes[index]):

            # -------------------------------------------
            #  Go through each person in the gallery
            # ------------------------------------------- #
            for person, pictures in self.gallery.items():
                #print("The number of pictures for " + str(person) + " is " + str(len(pictures)))

                nb_pred_diff = 0  # Nb times the person is predicted as different from the current probe

                # -------- Ensure balance in the gallery ------------
                pictures_gallery = get_balance_list(person, pictures, probe)

                # ------- Go through each picture of the current person of the gallery
                for _, picture in enumerate(pictures_gallery):
                    fr2_init = time.time()
                    fr_2 = picture.get_feature_repres(self.siamese_model) if not WITH_LATENT_REPRES else None
                    # print("The time to perform the fr2 is= " + str(time.time() - fr2_init))

                    # --- Go through each (synthetic) picture representing the probe --- #
                    for j, pict_probe in enumerate(probe.pictures):

                        fr_1 = pict_probe.get_feature_repres(self.siamese_model) if not WITH_LATENT_REPRES else None

                        # --- Distance reasoning for prediction ----
                        dist = picture.get_dist(probe.index[j], pict_probe, fr_1)

                        if DETAILED_PRINT:
                            picture.display_im(to_print="The compared face is printed and the dist is: " + str(dist))

                        if person not in probe.dist_pers:
                            probe.dist_pers[person] = []
                        probe.dist_pers[person].append(dist)

                        # --- Classification reasoning for prediction ----
                        if WITH_VOTE and self.siamese_model is not None:
                            same = self.siamese_model.output_from_embedding(fr_1, fr_2)
                            self.acc_model[1] += 1

                            # Check if "useful" to carry on
                            if same == 1:
                                if pict_probe.person != person:
                                    self.acc_model[0] += 1
                                    self.neg_recall[0] += 1
                                    self.neg_recall[1] += 1
                                else:
                                    self.pos_recall[1] += 1
                                    # print("Mistake: " + str(pict_probe.person) + " not predicted as " + str(person))
                                    #pict_probe.display_im(save=str(index) + str(i) + "_probe_" + pict_probe.person)
                                    #picture.display_im(save=str(index) + str(i) + "_gall_" + person)

                                nb_pred_diff += 1
                                if TOLERANCE < nb_pred_diff:
                                    break
                            else:
                                if pict_probe.person == person:
                                    self.pos_recall[1] += 1
                                    self.pos_recall[0] += 1
                                    self.acc_model[0] += 1

                                else:
                                    self.neg_recall[1] += 1
                                    # print("Mistake: " + str(pict_probe.person) + " predicted as " + str(person))
                                    #pict_probe.display_im(save=str(index) + str(i) + "_probe_" + pict_probe.person)
                                    #picture.display_im(save=str(index) + str(i) + "_gall_" + person)

                                probe.vote_pers[person] = 1 if person not in probe.vote_pers else probe.vote_pers[
                                                                                                      person] + 1

                # --- Distance reasoning for prediction ----
                # probe.avg_dist(person, len(pictures))
                probe.median_dist(person, len(pictures))

            # Predicted Person with class prediction reasoning
            if WITH_VOTE and self.siamese_model is not None:
                vote_init = time.time()
                probe.pred_from_vote(DETAILED_PRINT, res_vote)
                # print("The time to perform the vote is= " + str(time.time() - vote_init))

            # Predicted Person with distance reasoning
            dist_init = time.time()
            probe.predict_from_dist(res_dist, res_dist_topn)
            # print("The time to perform the dist based is= " + str(time.time() - dist_init))

            # Computation of the nb of false positives and false negatives
            probe.compute_false()

        print("\n------------------------------------------------------------------")
        if self.siamese_model is not None:
            if WITH_VOTE:
                print("The computed Accuracy for the model is: " + str(self.acc_model[0]) + "/" + str(self.acc_model[1])
                      + " (" + str(round(100.0 * self.acc_model[0] / self.acc_model[1], 2)) + "%)")
                print("The Positive Recall for the model is: " + str(self.pos_recall[0]) + "/" + str(self.pos_recall[1])
                      + " (" + str(round(100.0 * self.pos_recall[0] / self.pos_recall[1], 2)) + "%)")
                print("The Negative Recall for the model is: " + str(self.neg_recall[0]) + "/" + str(self.neg_recall[1])
                      + " (" + str(round(100.0 * self.neg_recall[0] / self.neg_recall[1], 2)) + "%)")
                print("Report: " + str(res_vote["nb_correct"]) + " correct, " + str(res_vote["nb_mistakes"])
                      + " wrong, " + str(res_vote["nb_not_recognized"]) + " undefined recognitions")

        print("Report with Distance: " + str(res_dist["nb_correct_dist"]) + " correct and " +
              str(res_dist["nb_mistakes_dist"]) + " wrong")
        for i, res_topn in enumerate(res_dist_topn):
            print("Report with Distance with top-" + str(N[i]) + " metric: " + str(res_topn) + " correct and " +
                  str(NB_PROBES - res_topn) + " wrong")
        print("------------------------------------------------------------------\n")

        return res_vote, res_dist, res_dist_topn

    '''--------------------- compute_far_frr -------------------------------------
    The function computes the False Acceptance Rate and the False Rejection Rate 
    for each considered threshold, from the number of false negatives and false
    positives that were counted for each probe.  
    ------------------------------------------------------------------------ '''

    def compute_far_frr(self):

        fp = [0.0] * len(THRESHOLDS_LIST)  # "different pictures that were predicted as similar"
        fn = [0.0] * len(THRESHOLDS_LIST)  # "similar pictures that were predicted as different"
        p = 0
        n = 0

        for i, index in enumerate(self.k_considered):
            for j, probe in enumerate(self.probes[index]):
                for l, fp_probe in enumerate(probe.fp_list):
                    fp[l] += float(fp_probe)
                    fn[l] += float(probe.fn_list[l])
                p += self.nb_sim
                n += self.nb_dif

        far = [x / n for x in fp]
        frr = [x / p for x in fn]

        # ------- Graph Representation -------
        dic = {"far": far, "frr": frr}
        title = "FAR and FRR according to the threshold"
        multi_line_graph(dic, THRESHOLDS_LIST, title, x_label="threshold", y_label="Rate Value", save_name="eer")
        return print_eer(far, frr)


# ================================================================
#                    Functions
# ================================================================


'''------------------- print_eer --------------------------
The function computes and prints the equal error rate 
(i.e value of far once far=frr)
----------------------------------------------------------- '''


def print_eer(far, frr):
    i = 0
    try:
        while far[i] < frr[i]:
            i += 1
    except IndexError:
        print("The range of threshold is too small: No intersection between far and frr!\n")

    eer = far[i] + far[i + 1] / 2
    print("The equal error rate is: " + str(eer))
    return eer


'''-------------------------- get_gallery ---------------------------------
The function returns a gallery with size_gallery people from the database 
IN: db_source_list: list of db 
OUT: dic where the key is the name of a person and the value 
is a list of FaceImage objects
-------------------------------------------------------------------------- '''


def get_gallery(size_gallery, db_source_list):
    face_dic = {}
    if not WITH_LATENT_REPRES:
        # ---------------------------------------------------------------------------
        # COMMON CASE: the face recognition task is based on the Siamese Network
        # ---------------------------------------------------------------------------
        for i, db in enumerate(db_source_list):
            try:
                if not WITH_SYNTHETIC_DATA:
                    face_dic.update(pickle.load(open(FOLDER_DIC + "faceDic_" + db + ".pkl", "rb")))
                else:
                    face_dic.update(pickle.load(open(FOLDER_DIC + "faceDic_" + db + "with_synth.pkl", "rb")))

            except (FileNotFoundError, EOFError) as e:
                print("The file " + FOLDER_DB + db + ".zip coundn't be found ... \n")
                fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + db + ".zip")
                face_dic = fileset.order_per_personName(TRANS, save=db, max_nb_pict=NB_IM_PER_PERS,
                                                        min_nb_pict=NB_IM_PER_PERS,
                                                        with_synth=WITH_SYNTHETIC_DATA)

    else:

        # ------------------------------------------------------------------------------------------------
        # PARTICULAR CASE: the face recognition task is based on the encoding derived from Style GAN
        # For saving time, in this case, use all the encoding that were already computed
        # ------------------------------------------------------------------------------------------------
        filenames = os.listdir(DLATENT_DIR)  # Each fn being like: dbname__person__index.npy

        # Go through the folder containing all the z
        for i, fn in enumerate(filenames):

            personName = fn.split(SEPARATOR)[1]
            index = fn.split(SEPARATOR)[2]
            img = FaceImage(fn, None, pers=personName, i=index)

            # ---------------------- Order z per person ----------------------
            try:
                face_dic[personName].append(img)
            except KeyError:
                face_dic[personName] = [img]

    if NB_IM_PER_PERS is not None:
        face_dic = {label: pictures for label, pictures in face_dic.items() if NB_IM_PER_PERS <= len(pictures)}
        face_dic = {label: pictures[:NB_IM_PER_PERS] for label, pictures in face_dic.items()}

    # Return "size_gallery" people
    if not WITH_SYNTHETIC_DATA:
        people = list(face_dic)
        Random().shuffle(people)
        return {k: v for k, v in face_dic.items() if k in people[:size_gallery]}
    else:
        face_dic = put_synth_first(face_dic)
        return {k: face_dic[k] for i, k in enumerate(face_dic) if i < size_gallery}



""" -------------------- put_synth_first --------------------
This function creates a new dictionary from the input one
 where all the people having synthetic images are put first in
---------------------------------------------------------------- """


def put_synth_first(face_dic):
    # ----------------------------------------------------------
    # 1. Extract all items where the person has synth images
    # ----------------------------------------------------------

    people_with_synt = {}
    people_without_synt = {}

    for person, pictures in face_dic.items():
        with_synt = False
        for i, picture in enumerate(pictures):
            if picture.is_synth:
                with_synt = True
                break

        if with_synt:
            people_with_synt.update({person: pictures})
        else:
            people_without_synt.update({person: pictures})

    # ----------------------------------------------------------
    # 2. Build dic where the extracted items are put first
    # ----------------------------------------------------------
    people_with_synt.update(people_without_synt)
    return people_with_synt


'''---------------------------- get_balance_list ------------------------------------------------------
This function removes from the current pictures list related to the current person in the gallery:
    - all the pictures attached to the current probe if the current gallery person is the current probe
    - random pictures from the pictures_gall so that it contains the same number of pictures
--------------------------------------------------------------------------------------------------------- '''


def get_balance_list(person_gall, pictures_gall, probe):  # REM: May take time!!
    # -------- Ensure balance in the gallery ------------
    if person_gall == probe.person:
        indexes = list(probe.index)
    else:
        indexes = [j for j in range(len(pictures_gall))]
        random.shuffle(indexes)
        indexes = indexes[:NB_INST_PROBES]
    indexes.sort(reverse=True)

    pictures_gallery = pictures_gall

    for inst_index in range(NB_INST_PROBES):
        pictures_gallery = pictures_gallery[:indexes[inst_index]] + pictures_gallery[indexes[inst_index] + 1:]

    return pictures_gallery


"""
This function returns the avg distance separating 2 consecutive "second value" 
in a list of items 
"""


def get_avg_sep(ordered_items):
    dist = 0
    for i, (val1, val2) in enumerate(ordered_items):
        if i == len(ordered_items) - 1:
            break
        dist += ordered_items[i + 1][1] - val2

    return dist / (len(ordered_items) - 1)


'''-------------- get_index_synth_pers --------------------------
IN: list of FaceImage objects
OUT: String representing the index of the real picture synthetic
images were generated from 
--------------------------------------------------------------- '''


def get_index_synth_pers(faceIm_list):
    index_str = None
    for i, faceIm in enumerate(faceIm_list):
        # Check if the current picture is synthetic
        if faceIm.is_synth:
            index_str = faceIm.index.split("_")[0]

    for i, faceIm in enumerate(faceIm_list):
        if faceIm.index == index_str:
            return i, index_str


'''--------------------------- get_synth_pict -------------------------------------
This function returns the list of real + corresponding synthetic pictures 
and a list of one element corresponding to the index of the real selected picture
----------------------------------------------------------------------------------- '''


def get_synth_pict(faceIm_list):
    try:
        j, index_str = get_index_synth_pers(faceIm_list)
    except TypeError:
        print("ERR: No synthetic data could be found...\n")
        return

    indexes_probe = [j]
    probe_pict = [faceIm_list[j]]

    synt_pic_list = []

    for i, faceIm in enumerate(faceIm_list):
        # Check if the current picture is synthetic
        if faceIm.is_synth and faceIm.index.split("_")[0] == j:
            synt_pic_list.append(faceIm)

    probe_pict.extend(synt_pic_list)

    return probe_pict, indexes_probe


'''-------------- remove_synth_data ------------------------------
This function removes from the gallery all the synthetic pictures
IN: gallery: dictionary where the key is the name of the person
and the value is the list of their pictures
--------------------------------------------------------------- '''


def remove_synth_data(gallery):
    for person, pictures in gallery.items():
        gallery[person] = [picture for i, picture in enumerate(pictures) if not picture.is_synth]


'''-------------- remove_real_data ------------------------------
This function removes from the gallery all the real pictures
IN: gallery: dictionary where the key is the name of the person
and the value is the list of their pictures
--------------------------------------------------------------- '''


def remove_real_data(gallery):
    face_dic = {}
    for person, pictures in gallery.items():
        # for i, pict in enumerate(pictures):
        # print("\nName of pict is " + str(pict.file_path))
        # print("Is synth " + str(pict.is_synth))
        gallery[person] = [picture for i, picture in enumerate(pictures) if picture.is_synth]
        if 0 < len(gallery[person]):
            face_dic[person] = gallery[person]
    return face_dic


# ================================================================
#                    MAIN
# ================================================================

if __name__ == '__main__':

    test_id = 2
    #model = "models/dsgbrieven_filteredcfp_humFilteredlfw_filteredfaceScrub_humanFiltered_3245_1default_70_" \
            #"triplet_loss_nonpretrained.pt"
    #model = "models/dsgbrieven_filteredlfw_filtered_8104_1default_70_cross_entropy_pretautoencoder.pt"
    #model = "models/dscfp_humFilteredgbrieven_filteredlfw_filteredfaceScrub_humanFiltered_15880_1default_70_" \
            #"triplet_loss_pretautoencoder.pt"
    model = "models/dsgbrieven_filteredcfp_humFilteredlfw_filteredfaceScrub_humanFiltered_3731_1default_100_" \
            "triplet_loss_pretautoencoder.pt"

    # --------------------
    #       Test 2
    # --------------------
    if test_id == 2:

        size_gallery = [20, 50, 100, 200]  # Nb of people to consider 20, 50, 100, 200,
        db_source_list = ["cfp_humFiltered"] #"testdb_filtered", "faceScrub_humanFiltered"] #

        for i, SIZE_GALLERY in enumerate(size_gallery):

            print("\n--------- The size of the gallery is " + str(SIZE_GALLERY) + " -----")

            fr = FaceRecognition(model, db_source=db_source_list)
            print("--------- The effective size of the gallery is " + str(len(fr.gallery)) + " -----\n")
            print("--------- The effective nb of probes is is " + str(len(fr.probes)) + " -----\n")
d
            # ------- Accumulators Definition --------------
            acc_nb_correct = 0
            acc_nb_mistakes = 0
            acc_nb_correct_dist = 0
            acc_nb_mistakes_dist = 0
            acc_nb_corr_dist_topN = [[] for i in range(len(N))]

            t_init = time.time()

            # ------- Build NB_REPET probes --------------
            for rep_index in range(NB_REPET):
                res_vote, res_dist, res_dist_topN = fr.recognition(rep_index)

                acc_nb_correct += res_vote["nb_correct"]
                acc_nb_mistakes += res_vote["nb_mistakes"]
                acc_nb_correct_dist += res_dist["nb_correct_dist"]
                acc_nb_mistakes_dist += res_dist["nb_mistakes_dist"]
                for j, res_topn in enumerate(res_dist_topN):  acc_nb_corr_dist_topN[j].append(res_topn)

            for j, res_topn_list in enumerate(acc_nb_corr_dist_topN):
                std = 0
                for i, res_topn in enumerate(res_topn_list):
                    std += abs(sum(res_topn_list)/ NB_REPET-res_topn)
                acc_nb_corr_dist_topN[j] = (min(sum(res_topn_list) / NB_REPET, NB_PROBES), std/NB_REPET)

            print("\n ------------------------------ Global Report ---------------------------------")
            if WITH_VOTE:
                print("Report: " + str(acc_nb_correct / NB_REPET) + " correct, " + str(
                    acc_nb_mistakes / NB_REPET) + " wrong recognitions")

            print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
                  + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
            for i, res_topn in enumerate(acc_nb_corr_dist_topN):
                print("Report with Distance with top-" + str(N[i]) + " metric: " + str(res_topn) +
                      " correct and " + str(NB_PROBES - res_topn[0]) + " wrong")
            print(" -------------------------------------------------------------------------------\n")

            # ------ Print Time -------
            total_time = str(time.time() - t_init)
            print("The time for the recognition of " + str(
                NB_PROBES * NB_REPET) + " people is " + total_time)

            # ------ Store all Information in CSV -------
            eer = fr.compute_far_frr()
            perc_vote_success = str(100 * acc_nb_correct / (NB_PROBES * NB_REPET))
            perc_dist_success = str(100 * acc_nb_correct_dist / (NB_PROBES * NB_REPET))
            print("perc_vote_success " + str(perc_vote_success) + " and perc_dist_success " + str(perc_dist_success))

            data = [NB_REPET, len(fr.gallery), NB_PROBES, NB_IM_PER_PERS, str(db_source_list)]

            acc = round(100.0 * fr.acc_model[0] / fr.acc_model[1], 2) if WITH_VOTE else 0
            recall = round(100.0 * fr.pos_recall[0] / fr.pos_recall[1], 2) if WITH_VOTE else 0
            algo = [model.split("models/")[1], acc, recall, TOLERANCE, WITH_LATENT_REPRES,
                    DIST_METRIC, NB_INST_PROBES, WITH_SYNTHETIC_DATA]

            result = [perc_vote_success, perc_dist_success, acc_nb_corr_dist_topN, eer, total_time]

            fr_in_csv(data, algo, result)
