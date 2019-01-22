from random import shuffle, randint
import torch
from Dataprocessing import from_zip_to_data
from Main import TRANS, NAME_MODEL, DEVICE, WITH_PROFILE
from TrainAndTest import oneshot

# ================================================================
#                   GLOBAL VARIABLES
# ================================================================

NB_TEST_PICT = 50
NB_PEOPLE = 400  # Nb of people to consider
TOLERANCE = 2  # Max nb of times the model can make mistake in comparing p_test and the pictures of 1 person

# ================================================================
#                    CLASS: FaceRecognition
# ================================================================


class FaceRecognition:

    def __init__(self, model_path):
        fileset = from_zip_to_data(WITH_PROFILE)
        people_dic = fileset.order_per_personName(TRANS, nb_people=NB_PEOPLE)
        self.people_pictures = people_dic

        # Pick different people (s.t. the test pictures are related to different people)
        people_test = list(people_dic.keys())
        shuffle(people_test)

        self.pictures_test = [] # list of tuples (person, picture)
        for i, person in enumerate(people_test[:NB_TEST_PICT]):
            j = randint(0, len(people_dic[person]) - 1)
            self.pictures_test.append((person, people_dic[person].pop(j)))

        model = torch.load(model_path)
        self.siamese_model = model

    '''---------------- recognition ---------------------------
     This function identifies the person on each test picture
     ----------------------------------------------------------'''
    def recognition(self):

        nb_not_recognized = 0
        nb_mistakes = 0
        nb_correct = 0
        detailed_print = False

        # --- Go through each test picture --- #
        for i, test_picture in enumerate(self.pictures_test):
            data = []
            nb_pred_diff = 0
            predictions = {}  # dict where the key is the person's name and the value the nb of "same" predictions

            data.append(torch.unsqueeze(test_picture[1].trans_img, 0))

            # --- Go through each person --- #
            for person, pictures in self.people_pictures.items():

                # --- Go through each picture of the current person --- #
                for i, picture in enumerate(pictures):

                    data.append(torch.unsqueeze(picture.trans_img, 0))
                    same = oneshot(self.siamese_model, DEVICE, data)

                    # --- Check the result of the prediction ---
                    if same == 1:
                        nb_pred_diff += 1
                        if TOLERANCE < nb_pred_diff:
                            data.pop()
                            break
                    else:
                        predictions[person] = 1 if person not in predictions else predictions[person] + 1

                    data.pop()

            # --- Final result --- #
            if detailed_print:
                print("predictions are " + str(predictions))
                print("The current person is: " + str(test_picture[0]))

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

        print("\nReport: " + str(nb_correct) + " correct, " + str(nb_mistakes) + " wrong, " + str(nb_not_recognized) + " undefined recognitions\n")


if __name__ == '__main__':
    fr = FaceRecognition(NAME_MODEL)
    fr.recognition()

