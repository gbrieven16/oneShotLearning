from random import shuffle, randint
import torch
from Dataprocessing import from_zip_to_data, TRANS
from Main import WITH_PROFILE, load_model

# ================================================================
#                   GLOBAL VARIABLES
# ================================================================

NB_PROBES = 10
SIZE_GALLERY = 50  # Nb of people to consider
TOLERANCE = 2  # Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person
NB_REPET = 10  # Nb of times test is repeated (over different probes)


# ================================================================
#                    CLASS: FaceRecognition
# ================================================================


class FaceRecognition:
    def __init__(self, model_path):
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
            people_gallery = list(self.gallery.keys())
            print("People Gallery is " + str(people_gallery))
            # Pick different people (s.t. the test pictures are related to different people)
            shuffle(people_gallery)

            probes = []
            for pers_i, person in enumerate(people_gallery[:NB_PROBES]):
                j = randint(0, len(self.gallery[person]) - 1)
                probes.append({"person": person, "pict": self.gallery[person][j], "index_pict": j})
            self.probes.append(probes)

            # To ensure having the same nb of pictures per person
            not_probes = []
            for pers_j, person in enumerate(people_gallery[NB_PROBES:]):
                not_probes.append(person)
            self.not_probes.append(not_probes)

        #print("self.probes is " + str(self.probes))

    '''---------------- recognition ---------------------------
     This function identifies the person on each test picture
     ----------------------------------------------------------'''

    def recognition(self, index):

        nb_not_recognized = 0
        nb_mistakes = 0
        nb_correct = 0

        nb_mistakes_dist = 0
        nb_correct_dist = 0

        detailed_print = True

        # --- Go through each probe --- #
        for i, probe in enumerate(self.probes[index]):  # [[[person, picture, index], [person, picture, index]], [[
            nb_pred_diff = 0
            predictions = {}  # dict where the key is the person's name and the value the nb of "same" predictions
            distances = {}

            fr_1 = probe["pict"].get_feature_repres(self.siamese_model)
            if detailed_print: probe[1].display_im(to_print="The face to identify is: ")

            # --- Go through each person in the gallery --- #
            for person, pictures in self.gallery.items():

                # Ensure balance in the gallery
                if person in self.not_probes or person == probe["person"]:
                    j = probe["index_pict"] if person == probe["person"] else randint(0, len(self.gallery[person]) - 1)
                    pictures_gallery = pictures[:j] + pictures[j + 1:]
                else:
                    pictures_gallery = pictures

                # --- Go through each picture of the current person --- #
                for l, picture in enumerate(pictures_gallery):

                    if detailed_print: picture.display_im(to_print="The face which is compared is: ")
                    fr_2 = picture.get_feature_repres(self.siamese_model)

                    # --- Distance reasoning for prediction ----
                    dist = get_distance(fr_1, fr_2)
                    distances[person] = dist if person not in distances else distances[person] + dist

                    # --- Classification reasoning for prediction ----
                    same = self.siamese_model.output_from_embedding(fr_1, fr_2)
                    if same == 1:
                        if detailed_print: print("Predicted as different")
                        nb_pred_diff += 1
                        if TOLERANCE < nb_pred_diff:
                            break
                    else:
                        predictions[person] = 1 if person not in predictions else predictions[person] + 1

                # --- Distance reasoning for prediction ----
                if 1 < len(pictures):
                    distances[person] = distances[person] / len(pictures)

            # --- Final result --- #
            if detailed_print:
                print("predictions are " + str(predictions))
                print("The current person is: " + str(probe["person"]))

            # Predicted Person with distance reasoning
            pred_pers_dist = sorted(distances.items(), key=lambda x: x[1])[0][0]
            if probe["person"] == pred_pers_dist:
                nb_correct_dist += 1
            else:
                nb_mistakes_dist += 1

            # Predicted Person with class prediction reasoning
            if 0 < len(predictions):
                pred_person = max(predictions, key=lambda k: predictions[k])
                if detailed_print: print("The predicted person is: " + pred_person + "\n")
                if probe["person"] == pred_person:
                    nb_correct += 1
                else:
                    nb_mistakes += 1
            else:
                if detailed_print: print("The person wasn't recognized!\n")
                nb_not_recognized += 1
        print("\n------------------------------------------------------------------")
        print("Report: " + str(nb_correct) + " correct, " + str(nb_mistakes) + " wrong, " + str(
            nb_not_recognized) + " undefined recognitions")

        print("Report with Distance: " + str(nb_correct_dist) + " correct, " + str(nb_mistakes_dist) + " wrong, " + str(
            nb_not_recognized) + " undefined recognitions")
        print("------------------------------------------------------------------\n")

        return nb_correct, nb_mistakes, nb_correct_dist, nb_mistakes_dist


# ================================================================
#                    Functions
# ================================================================

'''--------------------- get_distance -------------------------------------
The function returns the avg of the elements resulting from the difference
between the 2 given feature representations
------------------------------------------------------------------------ '''


def get_distance(feature_repr1, feature_repr2):
    difference = torch.abs(feature_repr2 - feature_repr1)
    sum_el = 0
    for j, elem in enumerate(difference[0]):
        sum_el += float(elem)
    return sum_el / len(difference[0])


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

    # ------- Build NB_REPET probes --------------
    for rep_index in range(NB_REPET):
        nb_correct, nb_mistakes, nb_correct_dist, nb_mistakes_dist = fr.recognition(rep_index)
        acc_nb_correct += nb_correct
        acc_nb_mistakes += nb_mistakes
        acc_nb_correct_dist += nb_correct_dist
        acc_nb_mistakes_dist += nb_mistakes_dist

    # ------ Print the average over all the different tests -------
    print("\n ------------------------------ Global Report ---------------------------------")
    print("Report: " + str(acc_nb_correct / NB_REPET) + " correct, " + str(acc_nb_mistakes / NB_REPET)
          + " wrong recognitions")

    print("Report with Distance: " + str(acc_nb_correct_dist / NB_REPET) + " correct, "
          + str(acc_nb_mistakes_dist / NB_REPET) + " wrong recognitions")
    print(" -------------------------------------------------------------------------------\n")
