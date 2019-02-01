from random import shuffle, randint
import torch
from Dataprocessing import from_zip_to_data
from Main import TRANS, WITH_PROFILE, NAME_MODEL

# ================================================================
#                   GLOBAL VARIABLES
# ================================================================

NB_TEST_PICT = 10
NB_PEOPLE = 50  # Nb of people to consider
TOLERANCE = 2  # Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person


# ================================================================
#                    CLASS: FaceRecognition
# ================================================================


class FaceRecognition:
    def __init__(self, model_path):
        fileset = from_zip_to_data(WITH_PROFILE)
        people_dic = fileset.order_per_personName(TRANS, nb_people=NB_PEOPLE, max_nb_pictures=4)
        self.people_pictures = people_dic

        # Pick different people (s.t. the test pictures are related to different people)
        people_test = list(people_dic.keys())
        shuffle(people_test)

        self.pictures_test = []  # list of tuples (person, picture)
        for i, person in enumerate(people_test[:NB_TEST_PICT]):
            j = randint(0, len(people_dic[person]) - 1)
            self.pictures_test.append((person, people_dic[person].pop(j)))

        # To ensure having the same nb of pictures per person
        for i, person in enumerate(people_test[NB_TEST_PICT:]):
            j = randint(0, len(people_dic[person]) - 1)
            person, people_dic[person].pop(j)

        model = torch.load(model_path)
        self.siamese_model = model

    '''---------------- recognition ---------------------------
     This function identifies the person on each test picture
     ----------------------------------------------------------'''

    def recognition(self):

        nb_not_recognized = 0
        nb_mistakes = 0
        nb_correct = 0
        nb_mistakes_dist = 0
        nb_correct_dist = 0
        detailed_print = False

        # --- Go through each test picture --- #
        for i, test_picture in enumerate(self.pictures_test):
            nb_pred_diff = 0
            predictions = {}  # dict where the key is the person's name and the value the nb of "same" predictions
            distances = {}

            fr_1 = test_picture[1].get_feature_repres(self.siamese_model)
            if detailed_print: test_picture[1].display_im(to_print="The face to identify is: ")

            # --- Go through each person in the gallery --- #
            for person, pictures in self.people_pictures.items():

                # --- Go through each picture of the current person --- #
                for i, picture in enumerate(pictures):

                    if detailed_print: picture.display_im(to_print="The face which is compared is: ")
                    fr_2 = picture.get_feature_repres(self.siamese_model)

                    same = self.siamese_model.output_from_embedding(fr_1, fr_2)

                    dist = get_distance(fr_1, fr_2)
                    distances[person] = dist if person not in distances else distances[person] + dist

                    # --- Check the result of the prediction ---
                    if same == 1:
                        if detailed_print: print("Predicted as different")
                        nb_pred_diff += 1
                        if TOLERANCE < nb_pred_diff:
                            break
                    else:
                        predictions[person] = 1 if person not in predictions else predictions[person] + 1

                if 1 < len(pictures):
                    distances[person] = distances[person] / len(pictures)

            # --- Final result --- #
            if detailed_print:
                print("predictions are " + str(predictions))
                print("The current person is: " + str(test_picture[0]))

            # Predicted Person with distance reasoning
            pred_pers_dist = sorted(distances.items(), key=lambda x: x[1])[0][0]
            if test_picture[0] == pred_pers_dist:
                nb_correct_dist += 1
            else:
                nb_mistakes_dist += 1

            # Predicted Person with class prediction reasoning
            if 0 < len(predictions):
                pred_person = max(predictions, key=lambda k: predictions[k])
                if detailed_print: print("The predicted person is: " + pred_person + "\n")
                if test_picture[0] == pred_person:
                    nb_correct += 1
                else:
                    nb_mistakes += 1
            else:
                if detailed_print: print("The person wasn't recognized!\n")
                nb_not_recognized += 1

        print("\nReport: " + str(nb_correct) + " correct, " + str(nb_mistakes) + " wrong, " + str(
            nb_not_recognized) + " undefined recognitions")

        print("Report with Distance: " + str(nb_correct_dist) + " correct, " + str(nb_mistakes_dist) + " wrong, " + str(
            nb_not_recognized) + " undefined recognitions\n")


def get_distance(feature_repr1, feature_repr2):
    difference = torch.abs(feature_repr2 - feature_repr1)
    sum_el = 0
    for j, elem in enumerate(difference[0]):
        sum_el += float(elem)
    return sum_el / len(difference[0])


if __name__ == '__main__':
    fr = FaceRecognition(NAME_MODEL)
    fr.recognition()
