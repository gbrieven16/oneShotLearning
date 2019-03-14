import time
from random import shuffle, randint
import numpy as np
import torch
from torch import nn
from Dataprocessing import from_zip_to_data, TRANS
from Visualization import multi_line_graph
from Main import WITH_PROFILE, load_model

# ================================================================
#                   GLOBAL VARIABLES
# ================================================================

NB_PROBES = 30
SIZE_GALLERY = 80  # Nb of people to consider
TOLERANCE = 3  # Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person
NB_REPET = 10  # Nb of times test is repeated (over different probes)
THRESHOLDS_LIST = list(np.arange(0, 5, 0.05))  # For MeanSquare
DETAILED_PRINT = False
DIST_METRIC = "Cosine_Sym"  #"MeanSquare" #"Manhattan"


# ======================================================================
#                    CLASS: FaceRecognition
# ======================================================================


class Probe:
    def __init__(self, person, picture, index_pict):
        self.person = person
        self.picture = picture
        self.index = index_pict

        self.dist_pers = {} # dic where the key is the person's name and the value is the list of distances from probe
        self.dist_avg_pers = {}
        self.vote_pers = {}

        self.fn_list = [0] * len(THRESHOLDS_LIST)  # Number of false negative (i.e. incorrect rejection)
        self.fp_list = [0] * len(THRESHOLDS_LIST)  # Number of false positive (i.e incorrect acceptance)

    def avg_dist(self, person, nb_pictures):
        if 1 < nb_pictures:
            self.dist_avg_pers[person] = sum(self.dist_pers[person]) / nb_pictures

    def median_dist(self, person, nb_pictures):
        pass

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
    def __init__(self, model_path):

        self.k_considered = []
        self.distances = {}
        # -------- Model Loading ----------------
        model = load_model(model_path)
        self.siamese_model = model

        # ------- Get data (from MAIN_ZIP) --------------
        fileset = from_zip_to_data(WITH_PROFILE)
        self.probes = []  # list of 10 lists of lists (person, picture, index_pict)
        self.not_probes = []  # list of of 10 lists of people not in probe (so that one of their picture is missed)

        # ------- Build NB_REPET probes --------------
        for rep_index in range(NB_REPET):
            self.gallery = fileset.order_per_personName(TRANS, nb_people=SIZE_GALLERY, same_nb_pict=True)
            self.nb_sim = len(self.gallery[next(iter(self.gallery))]) - 1  # nb of pictures "similar" to a probe
            self.nb_dif = self.nb_sim * (len(self.gallery) - 1)  # nb of pictures "different" from a probe

            people_gallery = list(self.gallery.keys())
            # print("People Gallery is " + str(people_gallery))
            # Pick different people (s.t. the test pictures are related to different people)
            shuffle(people_gallery)

            probes_k = []
            for pers_i, person in enumerate(people_gallery[:NB_PROBES]):
                j = randint(0, len(self.gallery[person]) - 1)
                probes_k.append(Probe(person, self.gallery[person][j], j))

            self.probes.append(probes_k)

            # To ensure having the same nb of pictures per person
            not_probes_k = []
            for pers_j, person in enumerate(people_gallery[NB_PROBES:]):
                not_probes_k.append(person)
            self.not_probes.append(not_probes_k)

            # print("self.probes is " + str(self.probes))

    '''---------------- recognition ---------------------------
     This function identifies the person on each test picture
     ----------------------------------------------------------'''

    def recognition(self, index):

        self.k_considered.append(index)

        res_vote = {"nb_not_recognized": 0, "nb_mistakes": 0, "nb_correct": 0}
        res_dist = {"nb_mistakes_dist": 0, "nb_correct_dist": 0}

        # --- Go through each probe --- #
        for i, probe in enumerate(self.probes[index]):
            fr_1 = picture.get_feature_repres(self.siamese_model)

            if DETAILED_PRINT: probe.picture.display_im(to_print="The face to identify is: ")

            # --- Go through each person in the gallery --- #
            for person, pictures in self.gallery.items():

                nb_pred_diff = 0  # Nb times the person is predicted as diffent from the current probe

                # Ensure balance in the gallery
                if person in self.not_probes[index] or person == probe.person:
                    j = probe.index if person == probe.person else randint(0, len(self.gallery[person]) - 1)
                    pictures_gallery = pictures[:j] + pictures[j + 1:]
                else:
                    pictures_gallery = pictures

                # --- Go through each picture of the current person --- #
                for l, picture in enumerate(pictures_gallery):

                    fr_2 = picture.get_feature_repres(self.siamese_model)

                    # --- Distance reasoning for prediction ----
                    dist = picture.get_dist(probe.person, probe.index, probe.picture, self.siamese_model)
                    #dist = get_distance(fr_1, fr_2)
                    if DETAILED_PRINT:
                        picture.display_im(to_print="The compared face is printed and the dist is: " + str(dist))

                    if person not in probe.dist_pers:
                        probe.dist_pers[person] = []
                    probe.dist_pers[person].append(dist)

                    # --- Classification reasoning for prediction ----
                    same = self.siamese_model.output_from_embedding(fr_1, fr_2)

                    # Check if "useful" to carry on
                    if same == 1:
                        if DETAILED_PRINT: print("Predicted as different")
                        nb_pred_diff += 1
                        if TOLERANCE < nb_pred_diff:
                            break
                    else:
                        probe.vote_pers[person] = 1 if person not in probe.vote_pers else probe.vote_pers[person] + 1

                # --- Distance reasoning for prediction ----
                probe.avg_dist(person, len(pictures))

            # Predicted Person with class prediction reasoning
            probe.pred_from_vote(DETAILED_PRINT, res_vote)

            # Predicted Person with distance reasoning
            probe.predict_from_dist(res_dist)

            # Computation of the nb of false positives and false negatives
            probe.compute_false()

        print("\n------------------------------------------------------------------")
        print("Report: " + str(res_vote["nb_correct"]) + " correct, " + str(res_vote["nb_mistakes"]) + " wrong, " + str(
            res_vote["nb_not_recognized"]) + " undefined recognitions")

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
        print_err(far, frr)


# ================================================================
#                    Functions
# ================================================================

'''--------------------- get_distance -------------------------------------
The function returns the avg of the elements resulting from the difference
between the 2 given feature representations
------------------------------------------------------------------------ '''


def get_distance(feature_repr1, feature_repr2):

    if DIST_METRIC == "Manhattan":
        difference_sum = torch.sum(torch.abs(feature_repr2 - feature_repr1))

    elif DIST_METRIC == "MeanSquare":
        difference_sum = torch.sum((feature_repr1 - feature_repr2) ** 2)

    elif DIST_METRIC == "Cosine_Sym":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(feature_repr1, feature_repr2)
    else:
        print("ERR: Invalid Distane Metric")
        raise IOError

    return difference_sum / len(feature_repr1[0])


'''------------------- print_eer --------------------------
The function computes and prints the equal error rate 
(i.e value of far once far=frr)
----------------------------------------------------------- '''


def print_err(far, frr):
    i = 0
    while far[i] < frr[i]:
        i += 1

    eer = far[i] + far[i + 1] / 2
    print("The equal error rate is: " + str(eer))


# ================================================================
#                    MAIN
# ================================================================

if __name__ == '__main__':
    fr = FaceRecognition("models/siameseFace_ds0123456_diff_100_32_triplet_loss.pt")

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
    print("Report: " + str(acc_nb_correct / NB_REPET) + " correct, " + str(acc_nb_mistakes / NB_REPET)
          + " wrong recognitions")

    print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
          + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
    print(" -------------------------------------------------------------------------------\n")

    print("The time for the recognition of " + str(NB_PROBES*NB_REPET) + " people is " + str(time.time() - t_init))

    fr.compute_far_frr()
