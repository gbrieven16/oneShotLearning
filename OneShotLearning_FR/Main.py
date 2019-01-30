import pickle
import torch
import torchvision.transforms as transforms
from NeuralNetwork import TYPE_ARCH, AutoEncoder

from Dataprocessing import Face_DS, from_zip_to_data, DB_TO_USE, MAIN_ZIP
from TrainAndTest import train, test, oneshot, pretraining, train_nonpretrained, get_optimizer, MARGIN
from Visualization import store_in_csv, visualization_test, visualization_train

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001  # To control regularization
LOSS = "triplet_loss" # "cross_entropy"
OPTIMIZER = "Adam"  # Adagrad "SGD"

SAVE_MODEL = False
DO_LEARN = True
PRETRAINING = "autoencoder" #"triplet_loss"  #  for autoencoder and None if no pretrain

DB_TRAIN = None       # If None, the instances of the training and test sets belong to different BD
DIFF_FACES = True     # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

TRANS = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
used_db = MAIN_ZIP.split("/")[-1] if DB_TO_USE is None else "".join([str(i) for i, db in enumerate(DB_TO_USE) if db != ""])

NAME_MODEL = "models/siameseFace" + "_ds" + used_db + (
    "_diff_" if DIFF_FACES else "_same_") + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + ".pt"


#########################################
#       FUNCTION main                   #
#########################################

def main(loss_type=LOSS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    autoencoder = AutoEncoder(device=DEVICE).to(DEVICE)
    visualization = True

    # ----------------------------------------------
    # Definition of a training and a testing set
    # ----------------------------------------------
    # Build your dataset from the processed data
    fileset = from_zip_to_data(WITH_PROFILE)
    training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES, db_train=DB_TRAIN)

    if DO_LEARN:
        # -----------------------
        #  training mode
        # -----------------------

        losses_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        acc_test = {"Pretrained Model": [], "Non-pretrained Model": []}

        face_train = Face_DS(training_set, transform=TRANS, device=DEVICE) # Triplet Version
        face_test = Face_DS(testing_set, transform=TRANS, device=DEVICE)   # Triplet Version
        train_loader = torch.utils.data.DataLoader(face_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(face_test, batch_size=batch_size, shuffle=False)

        if PRETRAINING is not None:
            # ---------- Pretraining using an autoencoder or classical model --------------
            train_data = Face_DS(training_set, transform=TRANS,
                                 device=DEVICE, triplet_version=False) if PRETRAINING is "autoencoder" else face_train

            model = pretraining(train_data, autoencoder, batch_size=batch_size, loss_type=PRETRAINING)
            train_nonpretrained(train_loader, test_loader, losses_test, acc_test, NUM_EPOCH, loss_type, OPTIMIZER)
        else:
            model = autoencoder.encoder

        # ------ Optimizer Definition ------
        optimizer = get_optimizer(model, OPTIMIZER, learning_rate, weight_decay)

        losses_train = []

        # ------- Model Training ---------
        for epoch in range(NUM_EPOCH):
            loss_list = train(model, DEVICE, train_loader, epoch, optimizer, loss_type=loss_type)
            loss, acc = test(model, DEVICE, test_loader, loss_type=loss_type)

            # Record for later visualization
            losses_train.append(loss_list)
            losses_test["Pretrained Model"].append(loss)
            acc_test["Pretrained Model"].append(acc)

            # --------- STOP if no relevant learning after some epoch ----------
            if 14 < epoch and sum(acc_test["Pretrained Model"]) / len(acc_test["Pretrained Model"]) < 55:
                print("The accuracy is bad => Stop Training")
                visualization = False
                break

        # ------- Model Saving ---------
        if SAVE_MODEL:
            torch.save(model, NAME_MODEL)
            with open(NAME_MODEL.split(".pt")[0] + '_testdata.pkl', 'wb') as output:
                pickle.dump(testing_set, output, pickle.HIGHEST_PROTOCOL)
            print("Model is saved!")

        # ------- Visualization: Evolution of the performance ---------
        if visualization:
            name_fig = "graphs/ds" + used_db + "_" + str(NUM_EPOCH) + "_" + str(batch_size) \
                       + "_" + loss_type + "_arch" + TYPE_ARCH
            visualization_train(range(0, NUM_EPOCH, int(round(NUM_EPOCH / 5))), losses_train,
                                save_name=name_fig + "_train.png")

            visualization_test(losses_test, acc_test, save_name=name_fig + "_test")

            # ------- Record: Evolution of the performance ---------
            info_data = [used_db, DIFF_FACES, WITH_PROFILE, DB_TRAIN]
            info_training = [PRETRAINING, NUM_EPOCH, batch_size, weight_decay, learning_rate,
                             TYPE_ARCH, OPTIMIZER, loss_type, MARGIN]
            info_result = [losses_test["Pretrained Model"], acc_test["Pretrained Model"]]
            store_in_csv(info_data, info_training, info_result)

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

        same = oneshot(model, data)
        if same == 0:
            print('=> PREDICTION: These two images represent the same person')
        else:
            print("=> PREDICTION: These two images don't represent the same person")


if __name__ == '__main__':
    main()
