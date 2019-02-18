import torch
import pickle
from functools import partial
from NeuralNetwork import TYPE_ARCH
from Model import Model, MARGIN, DEVICE

from Dataprocessing import Face_DS, from_zip_to_data, DB_TO_USE, MAIN_ZIP, CENTER_CROP, load_sets
from Visualization import store_in_csv
from Classification import Classifier

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001  # To control regularization
LOSS = "cross_entropy"  # "ce_classif" #"triplet_loss" "cross_entropy_with_cl" "constrastive_loss"
OPTIMIZER = "Adam"  # Adagrad "SGD"
WEIGHTED_CLASS = True

SAVE_MODEL = True
MODE = "learn"  # "classifier training"
PRETRAINING = "autoencoder"

DB_TRAIN = None  # If None, the instances of the training and test sets belong to different BD
DIFF_FACES = True  # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

used_db = MAIN_ZIP.split("/")[-1] if DB_TO_USE is None else "".join(
    [str(i) for i, db in enumerate(DB_TO_USE) if db != ""])

NAME_MODEL = "models/siameseFace" + "_ds" + used_db + (
    "_diff_" if DIFF_FACES else "_same_") + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + ".pt"


#########################################
#       FUNCTION main                   #
#########################################

def main(loss_type=LOSS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, wd=WEIGHT_DECAY, db_train=DB_TRAIN, fname=None):
    hyp_par = {"lr": lr, "wd": wd}
    train_param = {"loss_type": loss_type, "hyper_par": hyp_par, "opt_type": OPTIMIZER,
                   "weighted_class": WEIGHTED_CLASS}
    visualization = True

    fileset = from_zip_to_data(WITH_PROFILE, fname=fname)

    if MODE == "learn":
        # -----------------------
        #  training mode
        # -----------------------
        training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES, db_train=db_train)

        try:
            face_train, face_test = load_sets()
            print("The training and the testing sets have been loaded!")
        except IOError:  # FileNotFoundError:
            face_train = Face_DS(training_set, device=DEVICE, save="trainset_")  # Triplet Version
            face_test = Face_DS(testing_set, device=DEVICE, save="testset_")  # Triplet Version

        train_loader = torch.utils.data.DataLoader(face_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(face_test, batch_size=batch_size, shuffle=False)
        model = Model(train_param, train_loader=train_loader, test_loader=test_loader)

        if PRETRAINING == "autoencoder":
            # ---------- Pretraining using an autoencoder or classical model --------------
            model.pretraining(training_set, num_epochs=NUM_EPOCH, batch_size=batch_size)
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
            model.visualization(NUM_EPOCH, used_db, batch_size, OPTIMIZER)

            # ------- Record: Evolution of the performance ---------
            info_data = [used_db, DIFF_FACES, CENTER_CROP, DB_TRAIN]
            info_training = [PRETRAINING, NUM_EPOCH, batch_size, wd, lr,
                             TYPE_ARCH, OPTIMIZER, loss_type, WEIGHTED_CLASS, MARGIN]
            info_result = [model.losses_test["Pretrained Model"], model.f1_test["Pretrained Model"]]
            store_in_csv(info_data, info_training, info_result)

    elif MODE == "prediction":
        # --------------------------------------------
        #  prediction mode: 1 prediction
        # --------------------------------------------

        dataset = Face_DS(fileset, to_print=True, device=DEVICE)

        # batch_size = Nb of pairs you want to test
        prediction_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        model = torch.load(NAME_MODEL)

        # ------------ Data: list containing the tensor representations of the 2 images ---
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

    elif MODE == "evaluation":
        # -----------------------------------------------------------------
        #  Evaluation Mode (where MAIN_ZIP content is used for testing)
        # -----------------------------------------------------------------
        eval_network = load_model("models/siameseFace_ds0123456_diff_100_32_constrastive_loss.pt")
        eval_test = Face_DS(fileset, to_print=False, device=DEVICE)
        pred_loader = torch.utils.data.DataLoader(eval_test, batch_size=BATCH_SIZE, shuffle=True)

        eval_model = Model(test_loader=pred_loader, network=eval_network)
        print("Model Testing ...")
        eval_model.test()

    elif MODE == "classifier training":
        # -----------------------------------
        # Classification Approach (baseline)
        # -----------------------------------

        # Building of the dataset
        training_set, testing_set = fileset.get_train_and_test_sets(DIFF_FACES, db_train=db_train, classification=True)

        train_data = Face_DS(training_set, device=DEVICE, triplet_version=False)
        test_data = Face_DS(testing_set, device=DEVICE, triplet_version=False)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        # ------- Classifier Definition ---------
        classifier = Classifier(train_param=train_param, test_loader=test_loader, train_loader=train_loader,
                                nb_classes=train_data.nb_classes)

        if PRETRAINING == "autoencoder":
            # ---------- Pretraining using an autoencoder or classical model --------------
            classifier.pretraining(training_set, num_epochs=NUM_EPOCH, batch_size=batch_size)
            classifier.train_nonpretrained(NUM_EPOCH, optimizer_type=OPTIMIZER)

        # ------- Model Training ---------
        for epoch in range(NUM_EPOCH):
            classifier.train(epoch)
            classifier.test()


'''----------------------- load_model ----------------------------------------
 This function loads a model that was saved (under python 2.7)
 ---------------------------------------------------------------------------'''


def load_model(model_name):
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    return torch.load(model_name, map_location=lambda storage, loc: storage, pickle_module=pickle)  # network


if __name__ == '__main__':
    main()
    test = 2

    # -----------------------------------------------------------------------
    # Test 1: Confusion Matrix with different db for training and testing
    # -----------------------------------------------------------------------
    if test == 1:
        network = load_model("models/siameseFace_ds0123456_diff_100_32_constrastive_loss.pt")

        for i, filename in enumerate(['CASIA-WebFace.zip', "lfw.zip", "cfp.zip"]):
            fileset = from_zip_to_data(WITH_PROFILE, fname=MAIN_ZIP)
            testset = Face_DS(fileset, to_print=False, device=DEVICE)
            prediction_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

            model = Model(test_loader=prediction_loader, network=network)
            print("Model Testing on " + filename + " ...")
            model.test()

    # -----------------------------------------------------------------------
    # Test 2: Classification setting
    # -----------------------------------------------------------------------
    if test == 2:
        MODE = "classifier training"
        LOSS = "ce_classif"
        main()

    # -----------------------------------------------------------------------
    # "Test 3": Train Model from different db
    # -----------------------------------------------------------------------
    if test == 3:
        SAVE_MODEL = True
        db_train = ["cfp.zip", "testdb.zip", "testdb.zip"]
        for i, curr_db in enumerate(db_train):
            main(db_train=curr_db)



