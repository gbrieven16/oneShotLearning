import pickle
import torch
import torchvision.transforms as transforms
from NeuralNetwork import Net, TYPE_ARCH
from torch import optim

from Dataprocessing import Face_DS, MAIN_ZIP, from_zip_to_data, DB_TO_USE
from TrainAndTest import train, test, oneshot, visualization_test, visualization_train, LOSS
from Visualization import store_in_csv

#########################################
#       GLOBAL VARIABLES                #
#########################################


BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCH = 100
WEIGHT_DECAY = 0.001

SAVE_MODEL = True
DO_LEARN = True
DIFF_FACES = True  # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

TRANS = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])  # If applied, a dimensional error is raised

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
used_db = "".join([str(i) for i, db in enumerate(DB_TO_USE) if db != ""])

NAME_MODEL = "models/siameseFace" + "_ds" + used_db + (
    "_diff_" if DIFF_FACES else "_same_") + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + ".pt"


#########################################
#       FUNCTION main                   #
#########################################

def main():
    model = Net().to(DEVICE)

    # ----------------------------------------------
    # Definition of a training and a testing set
    # ----------------------------------------------
    # Build your dataset from the processed data
    fileset = from_zip_to_data(WITH_PROFILE)
    training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES)

    if DO_LEARN:

        # -----------------------
        #  training mode
        # -----------------------
        train_loader = torch.utils.data.DataLoader(Face_DS(training_set, transform=TRANS, device=DEVICE),
                                                   batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(Face_DS(testing_set, transform=TRANS, device=DEVICE),
                                                  batch_size=BATCH_SIZE, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        losses_train = []
        losses_test = []
        acc_test = []

        # ------- Model Training ---------
        for epoch in range(NUM_EPOCH):
            loss_list = train(model, DEVICE, train_loader, epoch, optimizer, BATCH_SIZE)
            loss, acc = test(model, DEVICE, test_loader)
            losses_train.append(loss_list)
            losses_test.append(loss)
            acc_test.append(acc)

        # ------- Model Saving ---------
        if SAVE_MODEL:
            torch.save(model, NAME_MODEL)
            with open(NAME_MODEL.split(".pt")[0] + '_testdata.pkl', 'wb') as output:
                pickle.dump(testing_set, output, pickle.HIGHEST_PROTOCOL)
            print("Model is saved!")

        # ------- Visualization: Evolution of the performance ---------
        name_fig = "ds" + used_db + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS
        visualization_train(range(0, NUM_EPOCH, round(NUM_EPOCH / 5)), losses_train, save_name=name_fig+"_train.png")
        visualization_test(losses_test, acc_test, save_name=name_fig+"_test.png")

        # ------- Record: Evolution of the performance ---------
        store_in_csv(BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, MAIN_ZIP, NUM_EPOCH, DIFF_FACES, WITH_PROFILE,
                     TYPE_ARCH, LOSS, losses_test, acc_test)

    else:
        # -----------------------
        #  prediction mode
        # -----------------------

        dataset = Face_DS(testing_set, transform=TRANS, to_print=True, device=DEVICE)

        # batch_size = Nb of pairs you want to test
        prediction_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        model = torch.load(NAME_MODEL)

        # ---------------------------------------------------------------------
        # Data: list containing the tensor representations of the 2 images
        # ---------------------------------------------------------------------
        data = []
        should_be_the_same = False

        if should_be_the_same:
            data.extend(next(iter(prediction_loader))[0][:2])  # The 2 faces are the same
            print("GROUNDTRUE: The 2 faces are the same ")

        else:
            data.extend(next(iter(prediction_loader))[0][:3:2])  # The 2 faces are different
            print("GROUNDTRUE: The 2 faces are different ")

        # print("One data given to the onshot function is: " + str(data[0]))

        same = oneshot(model, DEVICE, data)
        if same == 0:
            print('=> PREDICTION: These two images represent the same person')
        else:
            print("=> PREDICTION: These two images don't represent the same person")


if __name__ == '__main__':
    main()
