import time
import pickle
from functools import partial
import platform
import torch

# if platform.system() != "Darwin": torch.cuda.set_device(0)

from NeuralNetwork import TYPE_ARCH, NORMALIZE_FR, WITH_DIST_WEIGHT
from Model import Model, DEVICE, should_break, EP_SAVE
from Visualization import store_in_csv, line_graph

from Dataprocessing import Face_DS, from_zip_to_data, MAIN_ZIP, CENTER_CROP, load_sets, FOLDER_DB, TEST_ZIP, \
    WITH_SYNTH, FROM_ROOT, DIFF_FACES

#########################################
#       GLOBAL VARIABLES                #
#########################################


NUM_EPOCH = 1 if platform.system() == "Darwin" else 80
BATCH_SIZE = 32

LR_NONPRET = 0.001
LEARNING_RATE = 0.001
WITH_LR_SCHEDULER = "StepLR"
WEIGHT_DECAY = 0.001
OPTIMIZER = "Adam"  # "Adagrad" "SGD"

WEIGHTED_CLASS = False
WITH_EPOCH_OPT = False
LOSS = "triplet_loss"  # "ce_classif" "constrastive_loss" triplet_and_ce

MODE = "learn"  # "classifier training"
PRETRAINING = "none"  # "autoencoder" # "autoencoder_only" "none"
WITH_NON_PRET = True

WITH_PROFILE = False  # True if both frontally and in profile people

NB_PREDICTIONS = 1


##################################################################################################
#                                   FUNCTION main
# fname: list of the zip files to use to support the training and the validation sets (VS)
# db_train: list of the db to use for training (and then the other(s) are used to support the VS)
# name_model: if not None: retraining on and triplets built such that d(A,N) < d(A,P)
##################################################################################################

def main(db_train=None, fname=None, nb_classes=0, name_model=None, loss=LOSS):
    # -----------------------------------------
    # Train and Validation Sets Definition
    # -----------------------------------------
    if fname is not None:
        fileset = None
        for i, db in enumerate(fname):
            fileset = from_zip_to_data(WITH_PROFILE, fname=db, dataset=fileset)
    else:
        fileset = from_zip_to_data(WITH_PROFILE)

    # -----------------------------------------
    #  Test Set Definition
    # -----------------------------------------
    if loss != "ce_classif":
        fileset_test = from_zip_to_data(WITH_PROFILE, fname=TEST_ZIP)
        test_set, _ = fileset_test.get_sets(DIFF_FACES, db_set1=TEST_ZIP.split("/")[-1].split(".zip")[0])
    else:
        test_set = None

    if MODE == "learn":
        # =======================
        #  training mode
        # =======================

        # ------------------- Data Loading -----------------
        db_name, db_title = get_db_name(fname, db_train)
        training_set, validation_set = fileset.get_sets(DIFF_FACES, db_set1=db_train, nb_classes=nb_classes)
        model = None if name_model is None else load_model(name_model)
        sets_list = load_sets(db_name, DEVICE, nb_classes, [training_set, validation_set, test_set], model=model)

        # ------------------- Model Definition and Training  -----------------
        f1_score, model_name = main_train(sets_list, fname, db_train=db_train, name_model=name_model, loss=loss,
                                          nb_images=len(training_set.data_list))

        return f1_score

    elif MODE == "prediction":
        # ==============================================
        #  prediction mode: 1 prediction
        # ==============================================

        dataset = Face_DS(fileset=fileset, to_print=True)

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
        # =====================================================================
        #  Evaluation Mode (where MAIN_ZIP content is used for testing)
        # =====================================================================
        eval_network = load_model("models/siameseFace_ds0123456_diff_100_32_constrastive_loss.pt")
        eval_test = Face_DS(fileset=fileset, to_print=False)
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
    return torch.load(model_name, map_location=lambda storage, loc: storage, pickle_module=pickle).to(DEVICE)  # network


'''----------------------- get_db_name --------------------------------------
 This function returns a string merging the name of the different bd used 
 for training 
 ---------------------------------------------------------------------------'''


def get_db_name(fname, db_train, with_synt=WITH_SYNTH):
    if fname is None:
        db_name = MAIN_ZIP.split("/")[-1].split(".")[0]
    else:
        db_name = ""
        for i, db in enumerate(fname):
            if i != 0: db_name.join("_")
            db_name += (db.split("/")[-1].split(".")[0])

    db_title = "_".join(db_train) if db_train is not None else db_name
    if with_synt:
        db_title += "with_synth"
    return db_name, db_title


""" ------------------------------- main_train --------------------------------------
IN: sets_list: list of 3 Datasets (training, validation and testing)
------------------------------------------------------------------------------------- """


def main_train(sets_list, fname, db_train=None, name_model=None, scheduler=WITH_LR_SCHEDULER, pret=PRETRAINING,
               loss=LOSS, nb_images=0, with_synt=WITH_SYNTH):
    visualization = True
    num_epoch = NUM_EPOCH
    save_model = True
    db_name, db_title = get_db_name(fname, db_train, with_synt=with_synt)

    # -----------------------------------------
    # Define db_name as part of the file name
    # -----------------------------------------

    embeddingNet = None if name_model is None else load_model(name_model).embedding_net
    ds_info = "ds" + db_title + "_" + str(len(sets_list[0].train_data)) + "_"

    if with_synt:
        ds_info += "with_synt"

    if name_model is None:
        name_model = FROM_ROOT + "models/" + ds_info + TYPE_ARCH + "_" + str(
            NUM_EPOCH) + "_" + loss + "_pret" + pret + ".pt"

    # ----------------- Data Loaders definition ------------------------
    train_loader = torch.utils.data.DataLoader(sets_list[0], batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(sets_list[1], batch_size=BATCH_SIZE, shuffle=False)

    if loss != "ce_classif" and 2 < len(sets_list):
        test_loader = torch.utils.data.DataLoader(sets_list[2], batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_loader = None
    # GPUtil.showUtilization()

    # ------------------- Model Definition -----------------
    hyp_par = {"opt_type": OPTIMIZER, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "lr_scheduler": scheduler,
               "num_epoch": NUM_EPOCH}
    train_param = {"loss_type": loss, "hyper_par": hyp_par, "weighted_class": WEIGHTED_CLASS}

    model_learn = Model(train_param=train_param, train_loader=train_loader, validation_loader=validation_loader,
                        test_loader=test_loader, nb_classes=sets_list[0].nb_classes, embedding_net=embeddingNet)

    time_init = time.time()

    # ------------------- Pretraining -----------------
    try:
        if pret in ["autoencoder", "autoencoder_only"]:
            # ---------- Pretraining using an autoencoder or classical model --------------
            model_learn.pretraining(sets_list[0], hyp_par, batch_size=BATCH_SIZE)

            if WITH_NON_PRET:
                hyp_par_nonPret = {"opt_type": OPTIMIZER, "lr": LR_NONPRET, "wd": WEIGHT_DECAY,
                                   "lr_scheduler": scheduler, "num_epoch": NUM_EPOCH}
                model_learn.train_nonpretrained(NUM_EPOCH, hyp_par_nonPret, save=name_model)

        # ------- Model Training ---------
        if pret != "autoencoder_only":
            for epoch in range(NUM_EPOCH):
                if embeddingNet is None:
                    print("\n------- Training with " + TYPE_ARCH + " architecture and " + loss + " loss ----------")
                else:
                    print("\n------- Retraining of model ----------")

                model_learn.train(epoch)
                # if WITH_EVAL_ON_TRAIN: model_learn.prediction(on_train=True) Included in active_learning
                model_learn.prediction()

                if epoch != 0 and epoch % EP_SAVE == 0:
                    torch.save(model_learn.network.state_dict(), name_model + "_{0:03d}.pwf".format(epoch))

                # --------- STOP if no relevant learning after some epoch ----------
                if should_break(model_learn.f1_validation["Pretrained Model"], epoch) or model_learn.active_learning():
                    visualization = True
                    save_model = False
                    # num_epoch = epoch # not ok if not pretrained is better because the graph is cut ...
                    break

        raise KeyboardInterrupt
    except KeyboardInterrupt:
        # -------- Model Testing ----------------
        model_learn.prediction(validation=False) \
            if loss != "ce_classif" and 2 < len(sets_list) else "None"

        # ------- Model Saving ---------
        if save_model:
            model_learn.save_model(name_model)

        # ------- Visualization: Evolution of the performance ---------
        if visualization:
            # ------- Record: Evolution of the performance ---------
            info_data = [db_name if fname is not None else MAIN_ZIP, str(nb_images), len(sets_list[0].train_data),
                         DIFF_FACES, CENTER_CROP, db_title]
            if loss in ["triplet_loss", "contrastive_loss", "triplet_and_ce"]:
                loss += "_normFeat" + str(NORMALIZE_FR) + "_distWeight" + str(WITH_DIST_WEIGHT)
            info_training = [pret, NUM_EPOCH, BATCH_SIZE, WEIGHT_DECAY, str((LEARNING_RATE, scheduler)),
                             TYPE_ARCH, OPTIMIZER, loss, WEIGHTED_CLASS]

            # ---------------------- info_result -------------------------------------------
            # all losses on validation (dic with "nonPret" and "pret")
            # all f1 on validation (dic with "nonPret" and "pret")
            # Tuple of f1_pos and f1_neg on validation (dic with "nonPret" and "pret")
            # Tuple of test result on validation (dic with "nonPret" and "pret")
            # ------------------------------------------------------------------------------
            info_result = [model_learn.losses_validation, model_learn.f1_validation,
                           model_learn.f1_detail, model_learn.pos_recall, model_learn.f1_test]

            x = store_in_csv(info_data, info_training, info_result, time.time() - time_init)

            model_learn.visualization(num_epoch, db_title, len(sets_list[0].train_data))

            return x, name_model

        return None, name_model


#########################################
#       MAIN                            #
#########################################

if __name__ == '__main__':

    test = 2 if platform.system() == "Darwin" else 2

    # -----------------------------------------------------------------------
    # Test 1: Confusion Matrix with different db for training and testing
    # -----------------------------------------------------------------------
    if test == 1 or test is None:
        network = load_model(FROM_ROOT + "models/siameseFace_ds0123456_diff_100_32_triplet_loss.pt")

        for i, filename in enumerate(["testdb.zip", "lfw.zip", "cfp.zip", "ds0123456.zip"]):
            fileset = from_zip_to_data(WITH_PROFILE, fname=FOLDER_DB + filename)
            testset = Face_DS(fileset=fileset, to_print=False)
            predic_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

            model = Model(test_loader=predic_loader, network=network)
            print("Model Testing on " + filename + " ...")
            model.prediction()

    if test == 2 or test is None:
        print("-------------------------------------------------------------------------")
        print("Test 2: Classification setting with evolving number of different classes")
        print("-------------------------------------------------------------------------")
        MODE = "learn"
        nb_classes_list = [5, 10, 50, 100, 200, 500]  # , 1000, 5000, 10000]
        f1 = []
        db_name_train = [FOLDER_DB + "gbrieven_filtered.zip",
                         FOLDER_DB + "lfw_filtered.zip"]  # "faceScrub", "lfw", "cfp", "gbrieven", "testdb"]
        for i, nb_classes in enumerate(nb_classes_list):
            f1.append(main(nb_classes=nb_classes, fname=db_name_train, loss="ce_classif"))

        line_graph(nb_classes_list, f1, "f1 measure according to the number of classes")

    if test == 3 or test is None:
        print("----------------------------------------------------------------------- ")
        print("MAIN: Test 3: Train Model from different db")
        print("----------------------------------------------------------------------- ")
        db_name_train = [FOLDER_DB + "gbrieven_filtered.zip",
                         FOLDER_DB + "lfw_filtered.zip"]  # "faceScrub", "lfw", "cfp", "gbrieven", "testdb"]
        # db_name_train = [FOLDER_DB + "cfp70.zip"]
        # db_name_train = [FOLDER_DB + "gbrieven_filtered.zip"]
        loss_list = ["triplet_loss"]  # , "cross_entropy"] #, "constrastive_loss"]
        for i, loss in enumerate(loss_list):
            main(fname=db_name_train, loss=loss)

    if test == 4:  # test is None:
        print("----------------------------------------------------------------------- ")
        print("MAIN: Test 4: Train Model from all db")
        print("-----------------------------------------------------------------------\n ")

        db_name_train = [FOLDER_DB + "cfp_humFiltered.zip", FOLDER_DB + "gbrieven_filtered.zip",
                         FOLDER_DB + "lfw_filtered.zip", FOLDER_DB + "faceScrub_humanFiltered.zip"]

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
    # Test 6: Test model
    # -----------------------------------------------------------------------
    if test == 6 or test is None:
        MODE = "prediction"
        name_model = FROM_ROOT + "models/THEmodel_VGG16_triplet_loss_ep8.pt"
        db_name_train = [FOLDER_DB + "cfp70.zip"]  # , FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
        # FOLDER_DB + "faceScrub.zip"]
        main(name_model=name_model)

    if test == 7:  # test is None:
        print("-----------------------------------------------------------------------")
        print("Test 7: Retrain the model which has been registered on triplets")
        print("        such that d(A,N) < d(A,P)")
        print("-----------------------------------------------------------------------\n")
        MODE = "learn"
        name_model = FROM_ROOT + "models/dsgbrieven_filteredlfw_filtered_25375_1default_80_triplet_loss_pretautoencoder.pt"
        db_name_train = [FOLDER_DB + "gbrieven_filtered.zip", FOLDER_DB + "lfw_filtered.zip"]
        main(name_model=name_model, fname=db_name_train, loss="triplet_loss")
