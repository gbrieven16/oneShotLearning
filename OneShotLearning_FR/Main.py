import torch
import time
import pickle
from functools import partial
from NeuralNetwork import TYPE_ARCH
from Model import Model, MARGIN, DEVICE

from Dataprocessing import Face_DS, from_zip_to_data, DB_TO_USE, MAIN_ZIP, CENTER_CROP, load_sets, FOLDER_DB, TEST_ZIP
from Visualization import store_in_csv

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001  # To control regularization
LOSS = "cross_entropy"  # "ce_classif" #"triplet_loss" "cross_entropy_with_cl" "constrastive_loss"
OPTIMIZER = "Adam"  # Adagrad "SGD"
WEIGHTED_CLASS = True
WITH_EPOCH_OPT = False

SAVE_MODEL = True
MODE = "learn"  # "classifier training"
PRETRAINING = None  # "autoencoder"  # "autoencoder_only"

DB_TRAIN = None  # If None, the instances of the training and test sets belong to the same BD

DIFF_FACES = True  # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

used_db = MAIN_ZIP.split("/")[-1] if DB_TO_USE is None else "".join(
    [str(i) for i, db in enumerate(DB_TO_USE) if db != ""])

NAME_MODEL = "models/" + "ds" + used_db + ("_diff_" if DIFF_FACES else "_same_") \
             + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + \
             ("_pretrain" + PRETRAINING if PRETRAINING is not None else "") + ".pt"


########################################################################
#                       FUNCTION main
# db_train is used in the case where several db are in the same zip file
########################################################################

def main(loss_type=LOSS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, wd=WEIGHT_DECAY, db_train=DB_TRAIN, fname=None):
    hyp_par = {"lr": lr, "wd": wd}
    train_param = {"loss_type": loss_type, "hyper_par": hyp_par, "opt_type": OPTIMIZER,
                   "weighted_class": WEIGHTED_CLASS}
    visualization = True

    fileset = from_zip_to_data(WITH_PROFILE, fname=fname)
    fileset_test = from_zip_to_data(WITH_PROFILE, fname=TEST_ZIP)
    test_set, _ = fileset_test.get_sets(DIFF_FACES, db_set1=TEST_ZIP.split("/")[-1].split(".zip")[0])

    if MODE == "learn":
        # -----------------------
        #  training mode
        # -----------------------

        # ------------------- Data Loading -----------------
        training_set, validation_set = fileset.get_sets(DIFF_FACES, db_set1=db_train)

        try:
            face_train, face_validation, face_test = load_sets(fname)
            print("The training, validation and testing sets have been loaded!")
        except (IOError, FileNotFoundError) as e: # EOFError
            if loss_type == "ce_classif":
                face_train = Face_DS(training_set, device=DEVICE, save="trainset_", triplet_version=False)
                face_validation = Face_DS(validation_set, device=DEVICE, save="validationset_", triplet_version=False)
                face_test = Face_DS(test_set, device=DEVICE, save="testset_", triplet_version=False)
            else:
                face_train = Face_DS(training_set, device=DEVICE, save="trainset_")  # Triplet Version
                face_validation = Face_DS(validation_set, device=DEVICE, save="validationset_")  # Triplet Version
                face_test = Face_DS(test_set, device=DEVICE, save="testset_")  # Triplet Version

        train_loader = torch.utils.data.DataLoader(face_train, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(face_validation, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(face_test, batch_size=batch_size, shuffle=False)

        # ------------------- Model Definition -----------------

        model_learn = Model(train_param, train_loader=train_loader, validation_loader=validation_loader,
                            test_loader=test_loader, nb_classes=face_train.nb_classes)

        time_init = time.time()

        # ------------------- Pretraining -----------------

        if PRETRAINING == "autoencoder" or PRETRAINING == "autoencoder_only":
            # ---------- Pretraining using an autoencoder or classical model --------------
            model_learn.pretraining(training_set, num_epochs=NUM_EPOCH, batch_size=batch_size)
            model_learn.train_nonpretrained(NUM_EPOCH, optimizer_type=OPTIMIZER)

        # ------- Model Training ---------
        if PRETRAINING != "autoencoder_only":
            for epoch in range(NUM_EPOCH):
                print("---------- Training with " + TYPE_ARCH + " ----------")
                model_learn.train(epoch, with_epoch_opt=WITH_EPOCH_OPT)
                model_learn.prediction()

                # --------- STOP if no relevant learning after some epoch ----------
                curr_avg_f1 = sum(model_learn.f1_validation["Pretrained Model"]) / len(
                    model_learn.f1_validation["Pretrained Model"])
                if False and 14 < epoch and curr_avg_f1 < 55:
                    print("The f1 measure is bad => Stop Training")
                    visualization = False
                    break

        # -------- Model Testing ----------------
        model_learn.prediction(validation=False)

        # ------- Model Saving ---------
        if SAVE_MODEL:
            name_model = "models/" + "ds" + db_train + ("_diff_" if DIFF_FACES else "_same_") \
                         + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + "_pretrain" + PRETRAINING + ".pt"
            model_learn.save_model(name_model)

        # ------- Visualization: Evolution of the performance ---------
        if visualization:
            model_learn.visualization(NUM_EPOCH, used_db, batch_size, OPTIMIZER)

            # ------- Record: Evolution of the performance ---------
            info_data = [used_db, len(fileset.data_list), len(face_train.train_data), DIFF_FACES, CENTER_CROP, DB_TRAIN]
            info_training = [PRETRAINING, NUM_EPOCH, batch_size, wd, lr,
                             TYPE_ARCH, OPTIMIZER, loss_type, WEIGHTED_CLASS, MARGIN]
            info_result = [model_learn.losses_validation["Pretrained Model"],
                           model_learn.f1_validation["Pretrained Model"]]
            store_in_csv(info_data, info_training, info_result, time.time() - time_init)

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
        eval_model.prediction()


'''----------------------- load_model ----------------------------------------
 This function loads a model that was saved (under python 2.7)
 ---------------------------------------------------------------------------'''


def load_model(model_name):
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    return torch.load(model_name, map_location=lambda storage, loc: storage, pickle_module=pickle)  # network


if __name__ == '__main__':
    # main()
    test = 4

    # -----------------------------------------------------------------------
    # Test 1: Confusion Matrix with different db for training and testing
    # -----------------------------------------------------------------------
    if test == 1:
        network = load_model("models/siameseFace_ds0123456_diff_100_32_triplet_loss.pt")

        for i, filename in enumerate(["testdb.zip", "lfw.zip", "cfp.zip", "ds0123456.zip", ]):
            fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + filename)
            testset = Face_DS(fileset, to_print=False, device=DEVICE)
            predic_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

            model = Model(test_loader=predic_loader, network=network)
            print("Model Testing on " + filename + " ...")
            model.test()

    # -----------------------------------------------------------------------
    # Test 2: Classification setting
    # -----------------------------------------------------------------------
    if test == 2:
        MODE = "learn"
        main(loss_type="ce_classif")

    # -----------------------------------------------------------------------
    # "Test 3": Train Model from different db
    # -----------------------------------------------------------------------
    if test == 3:
        SAVE_MODEL = True
        db_train = ["ds0123456", "cfp", "lfw", "testdb"]
        for i, curr_db in enumerate(db_train):
            main(fname=FOLDER_DB + curr_db + ".zip")

    # -----------------------------------------------------------------------
    # Test 4: Train embedding network using an autoencoder and directly test
    #         face recognition
    # -----------------------------------------------------------------------
    if test == 4:
        PRETRAINING = "autoencoder_only"
        main()
        # => Go in FaceRecognition to test

        # -----------------------------------------------------------------------
        # Test 5: Test New Architecture: VGG16
        # -----------------------------------------------------------------------
