import torch
import torchvision.transforms as transforms
from NeuralNetwork import Net
from torch import optim

from Dataset import Face_DS, MAIN_ZIP, from_zip_to_data
from TrainAndTest import train, test, oneshot, visualization_test, visualization_train

#########################################
#       GLOBAL VARIABLES                #
# < 20/01: LEARNING_RATE = 0.001        #
#   21/01: LEARNING_RATE = 0.0001       #
#########################################


BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCH = 100
WEIGHT_DECAY = 0.0001

SAVE_DATA_TRAINING = True
SAVE_MODEL = True
DO_LEARN = False
DIFF_FACES = True  # If true, we have different faces in the training and the testing set

TRANS = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])  # If applied, a dimensional error is raised

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NAME_MODEL = "siameseFace" + "_" + MAIN_ZIP.split("datasets/")[1].split(".zip")[0] + (
    "_diff_" if DIFF_FACES else "_same_") + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + ".pt"


#########################################
#       FUNCTION main                   #
#########################################

def main():
    model = Net().to(DEVICE)

    # ----------------------------------------------
    # Definition of a training and a testing set
    # ----------------------------------------------
    # Build your dataset from the processed data
    fileset = from_zip_to_data()
    training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES)

    if DO_LEARN:

        # -----------------------
        #  training mode
        # -----------------------
        train_loader = torch.utils.data.DataLoader(Face_DS(training_set, transform=TRANS),
                                                   batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(Face_DS(testing_set, transform=TRANS),
                                                  batch_size=BATCH_SIZE, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        losses_train = []
        losses_test = []
        acc_test = []

        for epoch in range(NUM_EPOCH):
            loss_list = train(model, DEVICE, train_loader, epoch, optimizer, BATCH_SIZE)
            loss, acc = test(model, DEVICE, test_loader)
            losses_train.append(loss_list)
            losses_test.append(loss)
            acc_test.append(acc)

        if SAVE_MODEL:
            torch.save(model, NAME_MODEL)  # + '{:03}' .format(epoch))
            print("Model is saved!")

            visualization_train(range(0, NUM_EPOCH, round(NUM_EPOCH / 5)), losses_train)
            visualization_test(losses_test, acc_test)


    else:  # prediction
        dataset = Face_DS(testing_set, transform=TRANS, to_print=True)
        prediction_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                        shuffle=False)  # batch_size = Nb of pairs you want to test
        model = torch.load(NAME_MODEL)

        # ---------------------------------------------------------------------
        # Data: list containing the tensor representations of the 2 images
        # ---------------------------------------------------------------------
        data = []
        should_be_the_same = True

        if should_be_the_same:
            data.extend(next(iter(prediction_loader))[0][:2])  # The 2 faces are the same
            print("GROUNDTRUE: The 2 faces are the same ")

        else:
            data.extend(next(iter(prediction_loader))[0][:3:2])  # The 2 faces are different
            print("GROUNDTRUE: The 2 faces are different ")

        # print("One data given to the onshot function is: " + str(data[0]))

        same = oneshot(model, DEVICE, data)
        if same > 0:
            print('=> PREDICTION: These two images represent the same person')
        else:
            print("=> PREDICTION: These two images don't represent the same person")


if __name__ == '__main__':
    main()
