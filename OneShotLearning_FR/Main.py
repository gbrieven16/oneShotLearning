
import torch
import torchvision.transforms as transforms
from NeuralNetwork import TYPE_ARCH
from Model import Model, MARGIN, DEVICE

from Dataprocessing import Face_DS, from_zip_to_data, DB_TO_USE, MAIN_ZIP
from Visualization import store_in_csv

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001  # To control regularization
LOSS = "constrastive_loss" #"cross_entropy" _with_cl"  "cross_entropy" # "ce_and_tl"
OPTIMIZER = "Adam"  # Adagrad "SGD"
WEIGHTED_CLASS = True

SAVE_MODEL = True
DO_LEARN = True
PRETRAINING = None #"autoencoder" #None #

DB_TRAIN = None       # If None, the instances of the training and test sets belong to different BD
DIFF_FACES = True     # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

TRANS = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])

used_db = MAIN_ZIP.split("/")[-1] if DB_TO_USE is None else "".join([str(i) for i, db in enumerate(DB_TO_USE) if db != ""])

NAME_MODEL = "models/siameseFace" + "_ds" + used_db + (
    "_diff_" if DIFF_FACES else "_same_") + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + ".pt"


#########################################
#       FUNCTION main                   #
#########################################

def main(loss_type=LOSS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    hyp_par = {"lr": learning_rate, "wd": weight_decay}
    visualization = True

    # ----------------------------------------------
    # Definition of a training and a testing set
    # ----------------------------------------------

    fileset = from_zip_to_data(WITH_PROFILE)

    print("Training and Testing Sets Definition ... \n")
    training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES, db_train=DB_TRAIN)

    if DO_LEARN:
        # -----------------------
        #  training mode
        # -----------------------

        face_train = Face_DS(training_set, transform=TRANS, device=DEVICE) # Triplet Version
        face_test = Face_DS(testing_set, transform=TRANS, device=DEVICE)   # Triplet Version
        train_loader = torch.utils.data.DataLoader(face_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(face_test, batch_size=batch_size, shuffle=False)

        model = Model(train_loader, loss_type, tl=test_loader, hyper_par=hyp_par, opt_type=OPTIMIZER, weighted_class=WEIGHTED_CLASS)

        if PRETRAINING is not None:
            # ---------- Pretraining using an autoencoder or classical model --------------
            train_data = Face_DS(training_set, transform=TRANS, device=DEVICE, triplet_version=False)
            model.pretraining(train_data, num_epochs=NUM_EPOCH, batch_size=batch_size)
            model.train_nonpretrained(NUM_EPOCH, optimizer_type=OPTIMIZER)

        # ------- Model Training ---------
        for epoch in range(NUM_EPOCH):
            model.train(epoch)
            model.test()

            # --------- STOP if no relevant learning after some epoch ----------
            curr_avg_f1 = sum(model.f1_test["Pretrained Model"]) / len(model.f1_test["Pretrained Model"])
            if False and 14 < epoch and curr_avg_f1 < 55:
                print("The f1 measure is bad => Stop Training")
                visualization = False
                break

        # ------- Model Saving ---------
        if SAVE_MODEL: model.save_model(NAME_MODEL, testing_set)

        # ------- Visualization: Evolution of the performance ---------
        if visualization:

            model.visualization(NUM_EPOCH, used_db, batch_size)

            # ------- Record: Evolution of the performance ---------
            info_data = [used_db, DIFF_FACES, WITH_PROFILE, DB_TRAIN]
            info_training = [PRETRAINING, NUM_EPOCH, batch_size, weight_decay, learning_rate,
                             TYPE_ARCH, OPTIMIZER, loss_type, WEIGHTED_CLASS, MARGIN]
            info_result = [model.losses_test["Pretrained Model"], model.f1_test["Pretrained Model"]]
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

        same = model.network.predict(data)
        if same == 0:
            print('=> PREDICTION: These two images represent the same person')
        else:
            print("=> PREDICTION: These two images don't represent the same person")


if __name__ == '__main__':
    main()
