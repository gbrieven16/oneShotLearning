import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random
import time
import pickle
from random import shuffle, Random
import numpy as np

from Dataprocessing import from_zip_to_data, extract_randomly_elem, TRANS, FOLDER_DB, FaceImage, DIST_METRIC
from Visualization import multi_line_graph, fr_in_csv
from Main import WITH_PROFILE, load_model

# =====================================================================================================================
#                                   GLOBAL VARIABLES
# WITH_SYNTHETIC_DATA: if True, the synthetic images are used as instances of the probe to enforce the comparison
#                      (BUT not in the gallery)
# =====================================================================================================================

NB_PROBES = 20
SIZE_GALLERY = 80  # Nb of people to consider
TOLERANCE = 10  # 3 Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person
NB_REPET = 6  # Nb of times test is repeated (over different probes)
THRESHOLDS_LIST = list(np.arange(0, 5, 0.05))  # For MeanSquare
DETAILED_PRINT = False
NB_IM_PER_PERS = 10
WITH_SYNTHETIC_DATA = False
NB_INST_PROBES = 2

if WITH_SYNTHETIC_DATA:
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

    def avg_dist(self, person, nb_pictures):
        if 1 < nb_pictures:
            self.dist_avg_pers[person] = sum(self.dist_pers[person]) / nb_pictures

    def median_dist(self, person, nb_pictures):
        if 1 < nb_pictures:
            self.dist_avg_pers[person] = np.median(self.dist_pers[person])
            # print("median: " + str(self.dist_avg_pers[person]))

    def predict_from_dist(self, res_acc_dist):
        pred_pers_dist = sorted(self.dist_avg_pers.items(), key=lambda x: x[1])[0][0]
        if self.person == pred_pers_dist:
            res_acc_dist["nb_correct_dist"] += 1
        else:
            res_acc_dist["nb_mistakes_dist"] += 1

    def pred_from_vote(self, DETAILED_PRINT, res_vote):
        if DETAILED_PRINT:
            print("\n ------------- The current probe is: " + str(self.person) + " ----------------------- ")
            print("Voting system is represented by: " + str(self.vote_pers) + "\n")

        if 0 < len(self.vote_pers):
            pred_person = max(self.vote_pers, key=lambda k: self.vote_pers[k])
            if DETAILED_PRINT: print("The predicted person is: " + pred_person + "\n")
            if self.person == pred_person:
                res_vote["nb_correct"] += 1
            else:
                res_vote["nb_mistakes"] += 1
        else:
            if DETAILED_PRINT: print("The person wasn't recognized!\n")
            res_vote["nb_not_recognized"] += 1

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
    def __init__(self, model_path, db_source="testdb"):
        self.k_considered = []
        self.distances = {}

        # -------- Model Loading ----------------
        if model_path is not None:
            model = load_model(model_path)
            self.siamese_model = model

        # ------- Get data (from MAIN_ZIP) --------------
        self.probes = []  # list of NB_REPET lists of lists (person, picture, index_pict)

        # ------- Gallery Definition --------------
        self.gallery = get_gallery(SIZE_GALLERY, db_source)
        self.nb_sim = len(self.gallery[next(iter(self.gallery))]) - 1  # nb of pictures "similar" to a probe
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
                    probe_pict, indexes_probe = get_synth_pict(self.gallery[person])

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

        # --- Go through each probe --- #
        for i, probe in enumerate(self.probes[index]):

            # --- Go through each person in the gallery --- #
            for person, pictures in self.gallery.items():

                nb_pred_diff = 0  # Nb times the person is predicted as diffent from the current probe

                # -------- Ensure balance in the gallery ------------
                pictures_gallery = get_balance_list(person, pictures, probe)

                # print("\nSize pictures_gallery: " + str(len(pictures_gallery)) + " for person " + str(person))

                # --- Go through each picture of the current person of the gallery --- #
                for l, picture in enumerate(pictures_gallery):

                    fr_2 = picture.get_feature_repres(self.siamese_model)

                    # --- Go through each (synthetic) picture representing the probe --- #
                    for j, pict_probe in enumerate(probe.pictures):
                        fr_1 = pict_probe.get_feature_repres(self.siamese_model)

                        if DETAILED_PRINT: pict_probe.display_im(to_print="The face to identify is: ")

                        # --- Distance reasoning for prediction ----
                        dist = picture.get_dist(probe.person, probe.index[j], pict_probe, fr_1)
                        # dist = get_distance(fr_1, fr_2)
                        if DETAILED_PRINT:
                            picture.display_im(to_print="The compared face is printed and the dist is: " + str(dist))

                        if person not in probe.dist_pers:
                            probe.dist_pers[person] = []
                        probe.dist_pers[person].append(dist)

                        # --- Classification reasoning for prediction ----
                        if self.siamese_model is not None:
                            same = self.siamese_model.output_from_embedding(fr_1, fr_2)

                            # Check if "useful" to carry on
                            if same == 1:
                                if DETAILED_PRINT: print("Predicted as different")
                                nb_pred_diff += 1
                                if TOLERANCE < nb_pred_diff:
                                    break
                            else:
                                probe.vote_pers[person] = 1 if person not in probe.vote_pers else probe.vote_pers[
                                                                                                      person] + 1

                # --- Distance reasoning for prediction ----
                # probe.avg_dist(person, len(pictures))
                probe.median_dist(person, len(pictures))

            # Predicted Person with class prediction reasoning
            if self.siamese_model is not None:
                probe.pred_from_vote(DETAILED_PRINT, res_vote)

            # Predicted Person with distance reasoning
            probe.predict_from_dist(res_dist)

            # Computation of the nb of false positives and false negatives
            probe.compute_false()

        print("\n------------------------------------------------------------------")
        if self.siamese_model is not None:
            print("Report: " + str(res_vote["nb_correct"]) + " correct, " + str(res_vote["nb_mistakes"]) + " wrong, "
                  + str(res_vote["nb_not_recognized"]) + " undefined recognitions")

        print("Report with Distance: " + str(res_dist["nb_correct_dist"]) + " correct and " +
              str(res_dist["nb_mistakes_dist"]) + " wrong")
        print("------------------------------------------------------------------\n")

        return res_vote, res_dist

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
        multi_line_graph(dic, THRESHOLDS_LIST, title, x_label="threshold", y_label="Rate Value")
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
    while far[i] < frr[i]:
        i += 1

    eer = far[i] + far[i + 1] / 2
    print("The equal error rate is: " + str(eer))
    return eer


'''------------------- get_gallery --------------------------
The function returns a gallery with size_gallery people from
the database 
OUT: dic where the key is the name of a person and the value
is a list of FaceImage objects
----------------------------------------------------------- '''


def get_gallery(size_gallery, db_source):
    try:
        face_dic = pickle.load(open(FOLDER_DB + "faceDic_" + db_source + ".pkl", "rb"))
        if NB_IM_PER_PERS is not None:
            face_dic = {label: pictures for label, pictures in face_dic.items() if NB_IM_PER_PERS <= len(pictures)}
            face_dic = {label: pictures[:NB_IM_PER_PERS] for label, pictures in face_dic.items()}
    except FileNotFoundError:
        print("The file " + FOLDER_DB + db_source + ".zip coundn't be found ... \n")
        fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + db_source + ".zip")
        face_dic = fileset.order_per_personName(TRANS, save=db_source, max_nb_pict=NB_IM_PER_PERS,
                                                min_nb_pict=NB_IM_PER_PERS,
                                                with_synth=WITH_SYNTHETIC_DATA)

    # Return "size_gallery" people
    people = list(face_dic)
    Random().shuffle(people)
    return {k: v for k, v in face_dic.items() if k in people[:size_gallery]}


'''---------------------------- get_balance_list ------------------------------------------------------
This function removes from the current pictures list related 
to the current person in the gallery:
    - all the pictures attached to the current probe if the current gallery person is the current probe
    - random pictures from the pictures_gall so that it contains the same number of pictures
--------------------------------------------------------------------------------------------------------- '''


def get_balance_list(person_gall, pictures_gall, probe):
    # -------- Ensure balance in the gallery ------------
    if person_gall == probe.person:
        indexes = list(probe.index)
    else:
        indexes = [j for j in range(len(pictures_gall))]
        random.shuffle(indexes)
        indexes = indexes[:NB_INST_PROBES]
    indexes.sort(reverse=True)

    pictures_gallery = pictures_gall

    for i in range(NB_INST_PROBES):
        pictures_gallery = pictures_gallery[:indexes[i]] + pictures_gallery[indexes[i] + 1:]

    return pictures_gallery


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


'''------------------------- get_synth_pict --------------------------------
This function returns the list of real + corresponding synthetic pictures 
and a list of one element corresponding to the index of the real selected picture
--------------------------------------------------------------------------- '''


def get_synth_pict(faceIm_list):
    try:
        j, index_str = get_index_synth_pers(faceIm_list)
    except TypeError:
        print("ERR: No synthetic data could be found...\n")

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


# ================================================================
#                    MAIN
# ================================================================

if __name__ == '__main__':

    test_id = 2

    # --------------------
    #       Test 3
    # --------------------
    if test_id == 3:
        db_source = "cfp3"
        model = None
        fr = FaceRecognition(model, db_source=db_source)
        # ------- Accumulators Definition --------------
        acc_nb_correct_dist = 0
        acc_nb_mistakes_dist = 0

        # ------- Build NB_REPET probes --------------
        for rep_index in range(NB_REPET):
            res_vote, res_dist = fr.recognition(rep_index)

            acc_nb_correct_dist += res_dist["nb_correct_dist"]
            acc_nb_mistakes_dist += res_dist["nb_mistakes_dist"]

        # ------ Print the average over all the different tests -------
        print("\n ------------------------------ Global Report ---------------------------------")
        print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
              + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
        print(" -------------------------------------------------------------------------------\n")

        fr.compute_far_frr()

    # --------------------
    #       Test 1
    # --------------------
    if test_id == 1:
        db_source = "cfp3"
        model = "models/dsgbrievencfplfwfaceScrub_diff_100_32_triplet_loss_pretrainautoencoder.pt"
        fr = FaceRecognition(model, db_source=db_source)

        # ------- Accumulators Definition --------------
        acc_nb_correct = 0
        acc_nb_mistakes = 0
        acc_nb_correct_dist = 0
        acc_nb_mistakes_dist = 0

        # ------- Build NB_REPET probes --------------
        for rep_index in range(NB_REPET):
            res_vote, res_dist = fr.recognition(rep_index)

            acc_nb_correct += res_vote["nb_correct"]
            acc_nb_mistakes += res_vote["nb_mistakes"]
            acc_nb_correct_dist += res_dist["nb_correct_dist"]
            acc_nb_mistakes_dist += res_dist["nb_mistakes_dist"]

        # ------ Print the average over all the different tests -------
        print("\n ------------------------------ Global Report ---------------------------------")
        print("Report: " + str(acc_nb_correct / NB_REPET) + " correct, " + str(acc_nb_mistakes / NB_REPET)
              + " wrong recognitions")

        print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
              + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
        print(" -------------------------------------------------------------------------------\n")

        fr.compute_far_frr()

    # --------------------
    #       Test 2
    # --------------------
    if test_id == 2:

        size_gallery = [20, 50, 100, 200, 400]  # Nb of people to consider
        tolerance = [8, 5, 3]
        nb_im_per_pers = [None, 4, 8]
        db_source_list = ["cfp_humFiltered", "lfw_filtered", "gbrieven_filtered", "testdb_filtered", "faceScrub_filtered"]
        model = "models/dsgbrievencfplfwfaceScrub_diff_100_32_triplet_loss_pretrainautoencoder.pt"

        for k, db_source in enumerate(db_source_list):
            for i, SIZE_GALLERY in enumerate(size_gallery):
                for j, NB_IM_PER_PERS in enumerate(nb_im_per_pers):
                    for l, TOLERANCE in enumerate(tolerance):

                        print("Db source is: " + str(db_source))
                        print("The size of the gallery is " + str(SIZE_GALLERY))
                        print("The nb of images per person is " + str(NB_IM_PER_PERS))
                        print("The Tolerance is: " + str(TOLERANCE))

                        fr = FaceRecognition(model, db_source=db_source)

                        # ------- Accumulators Definition --------------
                        acc_nb_correct = 0
                        acc_nb_mistakes = 0
                        acc_nb_correct_dist = 0
                        acc_nb_mistakes_dist = 0
                        t_init = time.time()

                        # ------- Build NB_REPET probes --------------
                        for rep_index in range(NB_REPET):
                            res_vote, res_dist = fr.recognition(rep_index)

                            acc_nb_correct += res_vote["nb_correct"]
                            acc_nb_mistakes += res_vote["nb_mistakes"]
                            acc_nb_correct_dist += res_dist["nb_correct_dist"]
                            acc_nb_mistakes_dist += res_dist["nb_mistakes_dist"]

                        # ------ Print the average over all the different tests -------
                        print("\n ------------------------------ Global Report ---------------------------------")
                        print("Report: " + str(acc_nb_correct / NB_REPET) + " correct, " + str(
                            acc_nb_mistakes / NB_REPET)
                              + " wrong recognitions")

                        print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
                              + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
                        print(" -------------------------------------------------------------------------------\n")
                        total_time = str(time.time() - t_init)
                        print("The time for the recognition of " + str(
                            NB_PROBES * NB_REPET) + " people is " + total_time)

                        eer = fr.compute_far_frr()
                        perc_vote_success = str(acc_nb_correct / (NB_PROBES*NB_REPET))
                        perc_dist_success = str(acc_nb_correct_dist / (NB_PROBES*NB_REPET))

                        data = [NB_REPET, SIZE_GALLERY, NB_PROBES, NB_IM_PER_PERS, db_source]
                        algo = [model.split("models/")[1], TOLERANCE, DIST_METRIC, NB_INST_PROBES, WITH_SYNTHETIC_DATA]
                        result = [perc_vote_success, perc_dist_success, eer, total_time]
                        fr_in_csv(data, algo, result)
