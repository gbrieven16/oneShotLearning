import pickle
import random
import torch

from Main import load_model, WITH_PROFILE, DIFF_FACES, main_train
from Dataprocessing import FOLDER_DB, FaceImage, Face_DS, from_zip_to_data, TEST_ZIP, TRANS, FOLDER_DIC
from FaceRecognition import remove_synth_data, remove_real_data

#########################################
#       GLOBAL VARIABLES                #
#########################################

TEST_ID = 3
PICTURES_NB = [200, 500, 1000, 2000, 4000, 10000, 15000, 20000]
TRIPLET_NB = [5, 3, 2, 2, 2, 1, 1, 1]
NB_INST = 2
SIZE_VALIDATION_SET = 1000
SEED = 9
WITH_SYNTH = False

if TEST_ID == 6:
    PICTURES_NB = [3000]
    TRIPLET_NB = [1]
    WITH_SYNTH = True


#########################################
#       FUNCTIONS                       #
#########################################

"""
    - Take cfp & tested and , fix a certain nb of pictures (2000 typically). 
    - Sc1: take 500 people with 10 pictures => 5000 => 20000 pairs (2 triplets/pers)
    - Sc2: Take 1000 people with 5 pictures  => 5000 => 20000 pairs (2 triplets/pers)
    - Validation on lfw
"""


def build_ds(db_list, nb_people, nb_pict_per_pers):
    # 1. ------------- Order the picture per person ------------------------
    face_dic = {}
    for i, db in enumerate(db_list):
        try:
            face_dic.update(pickle.load(open(FOLDER_DB + FOLDER_DIC + "faceDic_" + db + ".pkl", "rb")))
            print("The dictionary storing pictures ordered by person has been loaded!\n")
        except FileNotFoundError:
            fileset = from_zip_to_data(False, fname=FOLDER_DB + db + ".zip")
            face_dic.update(fileset.order_per_personName(TRANS, save=db, with_synth=WITH_SYNTH, nb_people=nb_people))

    # 2. -------------- Shuffle all people ---------------------
    l = list(face_dic.items())
    random.shuffle(l)
    face_dic = dict(l)

    # 3. ------------- Impose the nb_pict_per_pers restriction ----------------
    same_nb_pict = {}
    for person, pictures in face_dic.items():
        if nb_pict_per_pers <= len(pictures):
            same_nb_pict[person] = pictures[:nb_pict_per_pers]

    # 4. ------------- Impose the nb people restriction ----------------------
    people_reduced = {}
    curr_nb_people = 0
    for person, pictures in same_nb_pict.items():
        curr_nb_people += 1
        if curr_nb_people < nb_people:
            people_reduced[person] = pictures

    # 5. ------------- Return Dataset from ------------------------
    return Face_DS(faces_dic=people_reduced)


""" -------------------- put_synth_first --------------------
This function creates a new dictionary from the input one
 where all the people having synthetic images are put first in
---------------------------------------------------------------- """


def put_synth_first(face_dic):
    # ----------------------------------------------------------
    # 1. Extract all items where the person has synth images
    # ----------------------------------------------------------

    people_with_synt = {}
    people_without_synt = {}

    for person, pictures in face_dic.items():
        with_synt = False
        for i, picture in enumerate(pictures):
            if picture.is_synth:
                with_synt = True
                break

        if with_synt:
            people_with_synt.update({person: pictures})
        else:
            people_without_synt.update({person: pictures})

    # ----------------------------------------------------------
    # 2. Build dic where the extracted items are put first
    # ----------------------------------------------------------
    people_with_synt.update(people_without_synt)
    return people_with_synt


"""
Training of models using different data quantities AND sources:
OUT: a list of 4 datasets if synth = True: [trainset_real, trainset_realSynth, validset, test_set]
                          else: [trainset_real, None, validset, test_set]
"""


def define_datasets(db_list, with_test_set=False, with_synt_first=False, with_synt=WITH_SYNTH, with_only_synth=False):
    name_ds_train_real = "dss_train_non_synth_q" + str(PICTURES_NB[-1])
    name_ds_val = "dss_validation"
    name_ds_train_withSynt = "dss_train_synth_q" + str(PICTURES_NB[-1])
    name_ds_train_onlySynt = "dss_train_onlysynth_q" + str(PICTURES_NB[-1])

    try:
        face_DS_list_real = load_model(FOLDER_DB + name_ds_train_real + ".pt")
        face_DS_valid = load_model(FOLDER_DB + name_ds_val + ".pt")
        face_DS_list_synt = load_model(FOLDER_DB + name_ds_train_withSynt + ".pt") if with_synt else []
        face_DS_list_only_synt = load_model(FOLDER_DB + name_ds_train_onlySynt + ".pt")
        print("The files content with loaders has been loaded ...\n")

    except FileNotFoundError:
        print("The files with loaders couldn't be found...\n")
        face_DS_list_real = None
        face_DS_valid = None
        face_DS_list_synt = None
        face_DS_list_only_synt = None

        # ------------------------------
        # 1. Load all the dictionaries
        # ------------------------------
        face_dic = {}

        for i, db in enumerate(db_list):
            try:
                face_dic.update(pickle.load(open(FOLDER_DB + FOLDER_DIC + "faceDic_" + db + ".pkl", "rb")))
                print("The dictionary storing pictures ordered by person has been loaded!\n")
            except (FileNotFoundError, EOFError) as e:
                print("The ordered pictures couldn't be found or loaded...\nBuilding... \n")
                fileset = from_zip_to_data(False, fname=FOLDER_DB + db + ".zip")
                face_dic.update(fileset.order_per_personName(TRANS, save=db, with_synth=with_synt))
        # ------------------------------
        # 2. Shuffle all people
        # ------------------------------
        l = list(face_dic.items())
        random.shuffle(l)
        face_dic = dict(l)

        # ------------------------------------------------------------
        # 3. Put first in the dictionary people having synth images
        # (REM: not necessary all the pictures but at least one)
        # ------------------------------------------------------------
        if with_synt_first:
            face_dic = put_synth_first(face_dic)

        # ------------------------------------------------------------
        # 4. From dictionary to dataset
        # ------------------------------------------------------------
        if not with_only_synth:
            face_DS_list_synt = from_dic_to_ds(face_dic, name_ds_train_withSynt, "real and synth") if with_synt else []
            remove_synth_data(face_dic)
            reduce_nb_inst(face_dic, NB_INST)
            face_DS_list_real = from_dic_to_ds(face_dic, name_ds_train_real, "real")
        else:
            face_dic_synth = remove_real_data(face_dic)
            face_DS_list_only_synt = from_dic_to_ds(face_dic_synth, name_ds_train_onlySynt, "synth")

        # ---------------------------------------------------------
        # 5. Take the rest of data to build the validation set
        # ---------------------------------------------------------
        face_dic = dict(list(face_dic.items())[:SIZE_VALIDATION_SET])
        face_DS_valid = Face_DS(faces_dic=face_dic, nb_triplet=1)

        # ----- Save Dataloader --------
        torch.save(face_DS_valid, FOLDER_DB + name_ds_val + ".pt")
        print("The validation dataset has been saved! \n")

    # ---------------------------------------------------------
    # 6. Define the testing set
    # ---------------------------------------------------------
    if with_test_set:
        name_test_ds = FOLDER_DB + "ds_testdb.pkl"
        try:
            with open(name_test_ds, "rb") as f:
                face_DS_test = torch.load(f)
                print('Test Set Loading Success!\n')
        except (ValueError, IOError) as e:  # EOFError  IOError FileNotFoundError
            fileset_test = from_zip_to_data(WITH_PROFILE, fname=TEST_ZIP)
            test_set, _ = fileset_test.get_sets(DIFF_FACES, db_set1=TEST_ZIP.split("/")[-1].split(".zip")[0])
            face_DS_test = Face_DS(fileset=test_set, triplet_version=True, save=name_test_ds)

        return [face_DS_list_real, face_DS_list_synt, face_DS_valid, face_DS_test, face_DS_list_only_synt]

    return [face_DS_list_real, face_DS_list_synt, face_DS_valid, face_DS_list_only_synt]


""" --------------------- from_dic_to_ds ----------------------------------- """


def from_dic_to_ds(face_dic, name_ds_train, nature):
    # Don't empty the dic supposed to support the real dataset just after
    face_dic_to_change = dict(face_dic) if nature == "real and synth" else face_dic

    # ---------------------------------------------------------------------------
    # Define "incremental" dictionaries (with 200, 300, 500... pictures in)
    # ---------------------------------------------------------------------------

    dict_list = []  # List of 8 dictionaries
    total_nb_pict = 0
    for j, nb_pict in enumerate(PICTURES_NB):
        dict_list.append({})
        while total_nb_pict < nb_pict:
            try:
                first_person = list(face_dic_to_change.keys())[0]
            except IndexError:
                print("The current final total nb of pictures is " + str(total_nb_pict) + "\n")
                break

            nb_synt, nb_real = get_nb_synth_data(list_pict=face_dic_to_change[first_person])
            total_nb_pict += nb_real
            dict_list[j][first_person] = face_dic_to_change.pop(first_person)

    # -------------------------------------------
    # Build Face_DS with images
    # -------------------------------------------
    face_DS_list = []
    for k, dic in enumerate(dict_list):
        face_DS_list.append(Face_DS(faces_dic=dic, nb_triplet=TRIPLET_NB[k]))

    # ----- Save Dataloader --------
    torch.save(face_DS_list, FOLDER_DB + name_ds_train + ".pt")
    print("The list of datasets with " + nature + " images only has been saved! \n")

    return face_DS_list


""" --------------------- merge_datasets -----------------------------------
This function merges the content of the datasets is list_ds and 
returns a single dataset Face_DS
------------------------------------------------------------------------ """


def merge_datasets(list_ds):
    all_ds = Face_DS()
    all_ds.merge_ds(list_ds)
    return all_ds


def get_nb_synth_data(db_path=None, list_pict=None):
    nb_synth = 0
    nb_real = 0

    if db_path is not None:
        fileset = from_zip_to_data(False, fname=db_path, dataset=None)
        list_pict = fileset.data_list

    for i, data in enumerate(list_pict):
        if data.synthetic:
            nb_synth += 1
        else:
            nb_real += 1

    print("The total number of real pictures is " + str(nb_real))
    print("The total number of synthetic pictures in is " + str(nb_synth))
    return nb_synth, nb_real


"""
This function removes some instances for each person in the 
number of instances is higher than the given nb_inst
"""


def reduce_nb_inst(face_dic, nb_inst, synth_appart=True):
    for person, pictures in face_dic.items():
        print("The size before reduction: " + str(len(pictures)))
        nb_synth, nb_real = get_nb_synth_data(list_pict=pictures)
        # ----------------------------------------
        # CASE 1: keep the synth pictures
        # ----------------------------------------
        if synth_appart and nb_inst < nb_real:
            filtered_pict_list = []
            nb_real_curr = 0
            for i, pict in enumerate(pictures):
                if pict.synthetic:
                    filtered_pict_list.append(pict)
                elif nb_real_curr < nb_inst:
                    filtered_pict_list.append(pict)
            face_dic[person] = filtered_pict_list

        # --------------------------------------------------------
        # CASE 2:No distinction between real and synth pictures
        # --------------------------------------------------------
        elif not synth_appart and nb_inst < nb_real + nb_synth:
            face_dic[person] = pictures[:nb_inst]
        print("The size after reduction: " + str(len(face_dic[person])))


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    # get_nb_synth_data(db_path=FOLDER_DB + "cfp_humFiltered.zip")
    test_id = TEST_ID

    db_list = ["gbrieven_filtered", "cfp_humFiltered", "lfw_filtered", "faceScrub_humanFiltered"]

    if test_id < 6:
        datasets = define_datasets(db_list, with_test_set=True, with_synt_first=True, with_synt=WITH_SYNTH)
    else:
        datasets = None
        db_list = ["cfp_humFiltered"]

    model_name = None
    db_name_train = [FOLDER_DB + "gbrieven_filtered.zip",
                     FOLDER_DB + "cfp_humFiltered.zip",
                     FOLDER_DB + "lfw_filtered.zip",
                     FOLDER_DB + "faceScrub_humanFiltered.zip"]

    if test_id == 1:
        # ================================ Tests Architectures and Losses ===============
        # 1. VGG16 & Triplet Loss
        # 2. VGG16 & Cross Entropy
        # 3. Basic Arch & Triplet Loss
        # 4. Basic Arch & Cross Entropy
        # 5. Basic Arch & Cross Entropy
        # => Choose best =================================================================
        index_ds = 5
        train_ds = merge_datasets(datasets[0][:index_ds])  # Explicit Reuse of previous data
        sets_list = [train_ds, datasets[2], datasets[3]]
        nb_pic = sum(PICTURES_NB[:index_ds])
        _, model_name = main_train(sets_list, db_name_train, name_model=model_name, nb_images=PICTURES_NB[nb_pic])

    if test_id == 2:
        # ============================================================
        #  With test Data Quantity (incremental approach)
        # !!!!! Hardcode nom de l'autoencoder (already trained)
        # ============================================================
        for i, nb_pict in enumerate(PICTURES_NB[:4]):
            print("================== Training on set " + str(i) + " with "
                  + str(sum(PICTURES_NB[:i + 1])) + " real images... ========================= ")
            train_ds = merge_datasets(datasets[0][:i + 1])  # Explicit Reuse of previous data
            sets_list = [train_ds, datasets[2], datasets[3]]
            main_train(sets_list, db_name_train, name_model=model_name, pret="autoencoder",
                       nb_images=nb_pict)  # "autoencoder"

        for i, nb_pict in enumerate(PICTURES_NB[:4]):
            print("=============== Training on set " + str(i) + " with "
                  + str(sum(PICTURES_NB[:i + 1])) + " real and synth images... ===================== ")
            train_ds = merge_datasets(datasets[1][:i + 1])  # Explicit Reuse of previous data
            sets_list = [train_ds, datasets[2], datasets[3]]
            main_train(sets_list, db_name_train, name_model=model_name, pret="autoencoder",
                       nb_images=nb_pict, with_synt=WITH_SYNTH)  # "autoencoder"

    if test_id == 3:
        # ============================================================
        #  With different Scheduler
        # ============================================================
        index_last = 4
        train_ds = merge_datasets(datasets[0][:index_last])  # Explicit Reuse of previous data

        sets_list = [train_ds, datasets[2], datasets[3]]
        print("len train set is " + str(len(sets_list[0].train_data)))
        schedulers = [None, "StepLR", "ExponentialLR"]
        for i, sched in enumerate(schedulers):
            try:
                print("================ Training with scheduler " + sched + "... ================================= ")
            except TypeError:
                print("================ Training without scheduler ... ================================= ")

            main_train(sets_list, db_name_train, scheduler=sched, pret="autoencoder",
                       nb_images=sum(PICTURES_NB[:index_last]))

    if test_id == 4:
        # ============================================================
        #  Wide vs Deep Dataset (particularity: nb epoch = 90)
        # ============================================================
        db = ["cfp_humFiltered", "testdb", "lfw_filtered"]
        print("-------------- Build Wide ds ... --------------------- ")
        wide_ds = build_ds(db, 500, 10)
        print("-------------- Build Deep ds ... --------------------- ")
        deep_ds = build_ds(db, 1000, 5)
        print("-------------- Build Validation ds ... --------------------- ")
        valid_ds = build_ds(["gbrieven_filtered"], 200, 4)
        print("-------------- Build test ds ... --------------------- ")
        test_ds = build_ds(["faceScrub_humanFiltered"], 200, 4)

        print("=============== TRAINING ON THE WIDE DS ===================== ")
        main_train([wide_ds, valid_ds, test_ds], db, pret="autoencoder", nb_images=5000)
        print("=============== TRAINING ON THE DEEP DS ===================== ")
        main_train([deep_ds, valid_ds, test_ds], db, pret="autoencoder", nb_images=5000)

    if test_id == 5:
        # ============================================================
        #  Check data quantity content in stored datasets_train.pt
        # ============================================================

        for i in range(len(PICTURES_NB)):
            print("======================= Check size on set " + str(i) + "... ================================= ")
            sets_list = [datasets[0][i], datasets[1], datasets[2]]
            print("Len dataset is " + str(len(sets_list[0].train_data)))

    if test_id == 6:
        # ============================================================
        #  Train only on synthetic data
        # ============================================================

        dss = define_datasets(db_list, with_test_set=True, with_only_synth=True)
        print("complete ds: " + str(dss))
        print("ds with only synthetic data: " + str(dss[-1]))
        sets_list = [dss[-1][0], dss[2], dss[3]]
        main_train(sets_list, db_name_train, pret="autoencoder", nb_images=PICTURES_NB[-1])
