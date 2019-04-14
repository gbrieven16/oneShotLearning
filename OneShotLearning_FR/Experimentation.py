import pickle
import random
import torch

from Main import main, BATCH_SIZE, load_model, WITH_PROFILE, DIFF_FACES, main_train
from Dataprocessing import FOLDER_DB, FaceImage, Face_DS, from_zip_to_data, TEST_ZIP, TRANS
from FaceRecognition import remove_synth_data

#########################################
#       GLOBAL VARIABLES                #
#########################################

PICTURES_NB = [200, 500, 1000, 2000, 4000, 10000, 15000, 20000]
TRIPLET_NB = [4, 2, 2, 2, 1, 1, 1, 1]
SIZE_VALIDATION_SET = 1000
SEED = 9

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
    # 1. ------------- Put db content in list ------------------------
    fileset = None
    for i, db in enumerate(db_list):
        fileset = from_zip_to_data(WITH_PROFILE, fname=db, dataset=fileset)

    # 2. ------------- Order picture/person ------------------------
    f_dic = fileset.order_per_personName(TRANS, nb_people=nb_people, max_nb_pict=nb_pict_per_pers, min_nb_pict=nb_pict_per_pers)

    # 3. ------------- Return Dataset from ------------------------
    return Face_DS(faces_dic=f_dic)


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
(HYP: min 1000 triplets to get some results) 

Total Quantity I can have: 28 595 - 688 = 27902 => less because min 2 pictures was imposed 
Real quantity of pictures: 21939 => Use 20 000 for training and the rest for validation 


Model1: Only real Picture: all filtered db with 200 pictures => 1000 triplets (2-3 triplets / person)
Model2: Only real Picture: all filtered db with 500 pictures => 1000 triplets (2 triplets / person)
Model3: Only real Picture: all filtered db with 1000 pictures => 4000 triplets (2 triplets / person)
Model4: Only real Picture: all filtered db with 2000 pictures => 8000 triplets (2 triplets / person)
Model5: Only real Picture: all filtered db with 4000 pictures => 8000 triplets (1 triplet / person)
Model6: Only real Picture: all filtered db with 8000 pictures => 16000 triplets (1 triplet / person)

Generate 1 big ds from *_filtered containing all data => Open all face_dics
Mix all people 

Take about 30 people (+200 pictures) and build 1000 triplets from 
Train on
Save the model => Model1

Take about 45 people (+300 pictures) and build 1000 triplets from 
Train Model1 on 
Save the model => Model2 

Take about 80 people (+500 pictures) and build 4000 triplets from 
Train Model2 on 
Save the model => Model3
... 
"""


def define_datasets(db_list, with_test_set=False, with_synt_first=False, with_synth=False):
    name_ds_train = "datasets_train"
    name_ds_val = "datasets_validation"

    try:
        face_DS_list = load_model(FOLDER_DB + name_ds_train + ".pt")
        face_DS_valid = load_model(FOLDER_DB + name_ds_val + ".pt")
        print("The files content with loaders has been loaded ...\n")

    except FileNotFoundError:

        print("The files with loaders couldn't be found...\n")

        # ------------------------------
        # 1. Load all the dictionaries
        # ------------------------------
        face_dic = {}

        for i, db in enumerate(db_list):
            try:
                face_dic.update(pickle.load(open(FOLDER_DB + "faceDic_" + db + ".pkl", "rb")))
            except FileNotFoundError:
                fileset = from_zip_to_data(False, fname=FOLDER_DB + db + ".zip")
                face_dic = fileset.order_per_personName(TRANS, save=db)

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

        if not with_synth:
            remove_synth_data(face_dic)

        # ---------------------------------------------------------------------------
        # 4. Define "incremental" dictionaries (with 200, 300, 500... pictures in)
        # ---------------------------------------------------------------------------

        dict_list = []  # List of 8 dictionaries
        total_nb_pict = 0
        for j, nb_pict in enumerate(PICTURES_NB):
            print("The current nb of pictures is " + str(nb_pict))
            print("The remaining nb of people is " + str(len(face_dic)))
            print("The current total nb of pictures is " + str(total_nb_pict) + "\n")
            dict_list.append({})
            while total_nb_pict < nb_pict:
                try:
                    first_person = list(face_dic.keys())[0]
                except IndexError:
                    print("The current final total nb of pictures is " + str(total_nb_pict) + "\n")
                    break
                total_nb_pict += len(face_dic[first_person])
                dict_list[j][first_person] = face_dic.pop(first_person)

        # -------------------------------------------
        # 5. Build Face_DS
        # -------------------------------------------

        face_DS_list = []
        for k, dic in enumerate(dict_list):
            face_DS_list.append(Face_DS(faces_dic=dic, nb_triplet=TRIPLET_NB[k]))

        # ----- Save Dataloader --------
        torch.save(face_DS_list, FOLDER_DB + name_ds_train + ".pt")
        print("The list of datasets has been saved! \n")

        # ---------------------------------------------------------
        # 6. Take the rest of data to build the validation set
        # ---------------------------------------------------------
        face_dic = dict(list(face_dic.items())[:SIZE_VALIDATION_SET])
        face_DS_valid = Face_DS(faces_dic=face_dic, nb_triplet=1)

        # ----- Save Dataloader --------
        torch.save(face_DS_valid, FOLDER_DB + name_ds_val + ".pt")
        print("The validation dataset has been saved! \n")

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

        return [face_DS_list, face_DS_valid, face_DS_test]

    return [face_DS_list, face_DS_valid]


def merge_datasets(list_ds):
    all_ds = Face_DS()
    all_ds.merge_ds(list_ds)
    return all_ds


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    db_list = ["cfp_humFiltered", "gbrieven_filtered", "lfw_filtered", "faceScrub_humanFiltered"]
    datasets = define_datasets(db_list, with_test_set=True)

    model_name = None
    db_name_train = [FOLDER_DB + "gbrieven.zip", FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
                     FOLDER_DB + "faceScrub.zip"]

    test_id = 2

    if test_id == 1:
        # ================================ Tests Architectures and Losses ===============
        # 1. VGG16 & Triplet Loss
        # 2. VGG16 & Cross Entropy
        # 3. Basic Arch & Triplet Loss
        # 4. Basic Arch & Cross Entropy
        # 5. Basic Arch & Cross Entropy
        # => Choose best =================================================================
        sets_list = [datasets[0][3], datasets[1], datasets[2]]
        _, model_name = main_train(sets_list, db_name_train, name_model=model_name)

    if test_id == 2:
        # ============================================================
        #  With test Data Quantity (incremental approach)
        # !!!!! Hardcode nom de l'autoencoder (already trained)
        # ============================================================
        for i in range(len(PICTURES_NB)):
            print("======================= Training on set " + str(i+1) + "... ================================= ")
            sets_list = [datasets[0][i+1], datasets[1], datasets[2]]
            _, model_name = main_train(sets_list, db_name_train, name_model=model_name, pret="autoencoder")

    if test_id == 3:
        # ============================================================
        #  With different Scheduler
        # ============================================================
        sets_list = [merge_datasets(datasets[0]), datasets[1], datasets[2]]
        schedulers = [None, "StepLR", "ExponentialLR"]
        for i, sched in enumerate(schedulers):
            print("================ Training with scheduler " + sched + "... ================================= ")
            _, model_name = main_train(sets_list, db_name_train, name_model=model_name, scheduler=sched)

    if test_id == 4:
        # ============================================================
        #  Wide vs Deep Dataset
        # ============================================================
        db = [FOLDER_DB + "cfp_humFiltered.zip", FOLDER_DB + "testdb.zip"]
        wide_ds = build_ds(db, 500, 10)
        deep_ds = build_ds(db, 1000, 5)
        valid_ds = build_ds([FOLDER_DB + "gbrieven.zip"], 200, 4)

        print("=============== TRAINING ON THE WIDE DS ===================== ")
        main_train([wide_ds, valid_ds], db, pret="autoencoder")
        print("=============== TRAINING ON THE DEEP DS ===================== ")
        main_train([deep_ds, valid_ds], db, pret="autoencoder")



