import platform
import torch

if platform.system() != "Darwin": torch.cuda.set_device(2)
import time
import pickle
from functools import partial
from NeuralNetwork import TYPE_ARCH
from Model import Model, DEVICE, should_break

from Dataprocessing import Face_DS, from_zip_to_data, MAIN_ZIP, CENTER_CROP, load_sets, FOLDER_DB, TEST_ZIP
from Visualization import store_in_csv, line_graph

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 8 if platform.system() == "Darwin" else 100
BATCH_SIZE = 32

LEARNING_RATE = 0.001
WITH_LR_SCHEDULER = "StepLR"  # "StepLR"  #"ExponentialLR"  # "StepLR"
WEIGHT_DECAY = 0.001  # To control regularization
OPTIMIZER = "Adam"  # Adagrad "SGD"

WEIGHTED_CLASS = False
WITH_EPOCH_OPT = False
LOSS = "cross_entropy"  # "triplet_loss"   "ce_classif"   "constrastive_loss"

MODE = "learn"  # "classifier training"
PRETRAINING = "none"  # ""autoencoder"  # "autoencoder_only" "none"

DIFF_FACES = True  # If true, we have different faces in the training and the testing set
WITH_PROFILE = False  # True if both frontally and in profile people

NB_PREDICTIONS = 1

##################################################################################################
#                                   FUNCTION main
# fname: list of the zip files to use to support the training and the validation sets (VS)
# db_train: list of the db to use for training (and then the other(s) are used to support the VS)
##################################################################################################

def main(loss_type=LOSS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, db_train=None, wd=WEIGHT_DECAY,
         fname=None, nb_classes=0, name_model=None):
    hyp_par = {"opt_type": OPTIMIZER, "lr": lr, "wd": wd, "lr_scheduler": WITH_LR_SCHEDULER, "num_epoch": NUM_EPOCH}
    train_param = {"loss_type": loss_type, "hyper_par": hyp_par, "weighted_class": WEIGHTED_CLASS}

    # Define db_name as part of the file name
    if name_model is None:
        embeddingNet = None
        db_name, db_title = get_db_name(fname, db_train)
        name_model = "models/" + "ds" + db_title + ("_diff_" if DIFF_FACES else "_same_") \
                 + str(NUM_EPOCH) + "_" + str(BATCH_SIZE) + "_" + LOSS + "_pretrain" + PRETRAINING + ".pt"
    else:
        embeddingNet = load_model(name_model).embedding_net
        db_name, db_title = get_db_name(fname, db_train)

    visualization = True
    save_model = True

    # -------- Train and Validation Sets Definition --------
    if fname is not None:
        fileset = None
        for i, db in enumerate(fname):
            fileset = from_zip_to_data(WITH_PROFILE, fname=db, dataset=fileset)
    else:
        fileset = from_zip_to_data(WITH_PROFILE)

    # -------- Test Set Definition --------
    if loss_type != "ce_classif":
        fileset_test = from_zip_to_data(WITH_PROFILE, fname=TEST_ZIP)
        test_set, _ = fileset_test.get_sets(DIFF_FACES, db_set1=TEST_ZIP.split("/")[-1].split(".zip")[0])
    else:
        test_set = None

    if MODE == "learn":
        # -----------------------
        #  training mode
        # -----------------------

        # ------------------- Data Loading -----------------
        training_set, validation_set = fileset.get_sets(DIFF_FACES, db_set1=db_train, nb_classes=nb_classes)

        sets_list = load_sets(db_name, DEVICE, nb_classes, [training_set, validation_set, test_set])

        train_loader = torch.utils.data.DataLoader(sets_list[0], batch_size=batch_size, shuffle=True)

        validation_loader = torch.utils.data.DataLoader(sets_list[1], batch_size=batch_size, shuffle=False)

        if loss_type != "ce_classif":
            test_loader = torch.utils.data.DataLoader(sets_list[2], batch_size=batch_size, shuffle=False)
        else:
            test_loader = None
        # GPUtil.showUtilization()

        # ------------------- Model Definition -----------------
        model_learn = Model(train_param=train_param, train_loader=train_loader, validation_loader=validation_loader,
                            test_loader=test_loader, nb_classes=sets_list[0].nb_classes, embedding_net=embeddingNet)

        time_init = time.time()

        # ------------------- Pretraining -----------------

        if PRETRAINING == "autoencoder" or PRETRAINING == "autoencoder_only":
            # ---------- Pretraining using an autoencoder or classical model --------------
            model_learn.pretraining(sets_list[0], hyp_par, num_epochs=NUM_EPOCH, batch_size=batch_size)
            model_learn.train_nonpretrained(NUM_EPOCH, hyp_par)

        # ------- Model Training ---------
        if PRETRAINING != "autoencoder_only":
            for epoch in range(NUM_EPOCH):
                if embeddingNet is None:
                    print("\n------- Training with " + TYPE_ARCH + " architecture ----------")
                else:
                    print("\n------- Retraining of model ----------")

                model_learn.train(epoch, with_epoch_opt=WITH_EPOCH_OPT)
                model_learn.prediction()

                # --------- STOP if no relevant learning after some epoch ----------
                if should_break(model_learn.f1_validation["Pretrained Model"], epoch):
                    visualization = False
                    save_model = False
                    break

        # -------- Model Testing ----------------

        f1_test = model_learn.prediction(validation=False) if loss_type != "ce_classif" else "classif"

        # ------- Model Saving ---------
        if save_model:
            model_learn.save_model(name_model)

        # ------- Visualization: Evolution of the performance ---------
        if visualization:
            model_learn.visualization(NUM_EPOCH, db_title, batch_size, OPTIMIZER)

            # ------- Record: Evolution of the performance ---------
            info_data = [db_name if fname is not None else MAIN_ZIP, len(fileset.data_list),
                         len(sets_list[0].train_data),
                         DIFF_FACES, CENTER_CROP, db_title]
            info_training = [PRETRAINING, NUM_EPOCH, batch_size, wd, str((lr, WITH_LR_SCHEDULER)),
                             TYPE_ARCH, OPTIMIZER, loss_type, WEIGHTED_CLASS]
            info_result = [model_learn.losses_validation["Pretrained Model"],
                           model_learn.f1_validation["Pretrained Model"], str(f1_test)]
            return store_in_csv(info_data, info_training, info_result, time.time() - time_init)

    elif MODE == "prediction":
        # --------------------------------------------
        #  prediction mode: 1 prediction
        # --------------------------------------------

        dataset = Face_DS(fileset=fileset, to_print=True, device=DEVICE)

        model = load_model(name_model)

        # ------------ Data: list containing the tensor representations of the 2 images ---
        for i in range(NB_PREDICTIONS):
            print(" ---------------- Prediction " + str(i) + "----------------\n")
            # batch_size = Nb of pairs you want to test
            prediction_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
            data = []
            should_be_the_same = False

            if should_be_the_same:
                data.extend(next(iter(prediction_loader))[0][:2])  # The 2 faces are the same
                print("GROUNDTRUE: The 2 faces are the same ")
            else:
                print("next(iter(prediction_loader))[i][:3:2] " + str(next(iter(prediction_loader))[i][:3:2]))
                data.extend(next(iter(prediction_loader))[0][:3:2])  # The 2 faces are different
                print("GROUNDTRUE: The 2 faces are different ")

            # print("One data given to the onshot function is: " + str(data[0]))

            same = model.predict(data)
            if same == 0:
                print('=> PREDICTION: These two images represent the same person')
            else:
                print("=> PREDICTION: These two images don't represent the same person")

    elif MODE == "evaluation":
        # -----------------------------------------------------------------
        #  Evaluation Mode (where MAIN_ZIP content is used for testing)
        # -----------------------------------------------------------------
        eval_network = load_model("models/siameseFace_ds0123456_diff_100_32_constrastive_loss.pt")
        eval_test = Face_DS(fileset=fileset, to_print=False, device=DEVICE)
        pred_loader = torch.utils.data.DataLoader(eval_test, batch_size=BATCH_SIZE, shuffle=True)

        eval_model = Model(test_loader=pred_loader, network=eval_network)
        print("Model Testing ...")
        eval_model.prediction()


'''----------------------- load_model ----------------------------------------
 This function loads a model that was saved (under python 2.7)
 ---------------------------------------------------------------------------'''


def load_model(model_name):
    pickle.load = partial(pickle.load)
    pickle.Unpickler = partial(pickle.Unpickler)
    return torch.load(model_name, map_location=lambda storage, loc: storage, pickle_module=pickle)  # network


'''----------------------- get_db_name --------------------------------------
 This function returns a string merging the name of the different bd used 
 for training 
 ---------------------------------------------------------------------------'''


def get_db_name(fname, db_train):
    if fname is None:
        db_name = MAIN_ZIP.split("/")[-1].split(".")[0]
    else:
        db_name = ""
        for i, db in enumerate(fname):
            if i != 0: db_name.join("_")
            db_name += (db.split("/")[-1].split(".")[0])

    return db_name, "_".join(db_train) if db_train is not None else db_name


#########################################
#       MAIN                            #
#########################################

if __name__ == '__main__':
    # main()
    test = 6 if platform.system() == "Darwin" else 4

    # -----------------------------------------------------------------------
    # Test 1: Confusion Matrix with different db for training and testing
    # -----------------------------------------------------------------------
    if test == 1 or test is None:
        network = load_model("models/siameseFace_ds0123456_diff_100_32_triplet_loss.pt")

        for i, filename in enumerate(["testdb.zip", "lfw.zip", "cfp.zip", "ds0123456.zip"]):
            fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + filename)
            testset = Face_DS(fileset=fileset, to_print=False, device=DEVICE)
            predic_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

            model = Model(test_loader=predic_loader, network=network)
            print("Model Testing on " + filename + " ...")
            model.prediction()

    # -------------------------------------------------------------------------
    # Test 2: Classification setting with evolving number of different classes
    # -------------------------------------------------------------------------
    if test == 2 or test is None:
        MODE = "learn"
        nb_classes_list = [5, 10, 50, 100, 200, 500, 1000, 5000, 10000]
        f1 = []
        db_name_train = [FOLDER_DB + "gbrieven.zip", FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
                         FOLDER_DB + "faceScrub.zip"]
        for i, nb_classes in enumerate(nb_classes_list):
            f1.append(main(loss_type="ce_classif", nb_classes=nb_classes, fname=db_name_train))

        line_graph(nb_classes_list, f1, "f1 measure according to the number of classes")

    # -----------------------------------------------------------------------
    # "Test 3": Train Model from different db
    # -----------------------------------------------------------------------
    if test == 3 or test is None:
        db_name_train = ["cfp70"]  # "faceScrub", "lfw", "cfp", "gbrieven", "testdb"] #"testCropped"
        for i, curr_db in enumerate(db_name_train):
            main(fname=[FOLDER_DB + curr_db + ".zip"])

    # -----------------------------------------------------------------------
    # "Test 4": Train Model from all db
    # -----------------------------------------------------------------------
    if test == 4 or test is None:
        print("Test 4: Training on all db ... \n")
        db_name_train = [FOLDER_DB + "gbrieven.zip", FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
                         FOLDER_DB + "faceScrub.zip"]
        main(fname=db_name_train)

    # -----------------------------------------------------------------------
    # Test 5: Train embedding network using an autoencoder and directly test
    #         face recognition
    # -----------------------------------------------------------------------
    if test == 5 or test is None:
        PRETRAINING = "autoencoder_only"
        main()
        # => Go in FaceRecognition to test

    # -----------------------------------------------------------------------
    # Test 6: Test the model which has been registered
    # -----------------------------------------------------------------------
    if test == 6 or test is None:
        MODE = "prediction"
        main(name_model="models/dsgbrievencfplfwfaceScrub_diff_100_32_triplet_loss_pretrainautoencoder.pt")

    # -----------------------------------------------------------------------
    # Test 7: Test the model which has been registered
    # -----------------------------------------------------------------------
    if test == 7 or test is None:
        MODE = "learn"
        name_model = "models/dsgbrievencfplfwfaceScrub_diff_100_32_triplet_loss_pretrainautoencoder.pt"
        db_name_train = [FOLDER_DB + "cfp.zip"] #, FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
                         #FOLDER_DB + "faceScrub.zip"]
        main(name_model=name_model, fname=db_name_train)