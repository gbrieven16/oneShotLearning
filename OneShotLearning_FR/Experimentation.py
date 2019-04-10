import pickle
import random
import torch

from Main import main, BATCH_SIZE, load_model, WITH_PROFILE, DIFF_FACES, main_train
from Dataprocessing import FOLDER_DB, FaceImage, Face_DS, from_zip_to_data, TEST_ZIP

#########################################
#       GLOBAL VARIABLES                #
#########################################

PICTURES_NB = [200, 500, 1000, 2000, 4000, 10000, 15000, 20000]
TRIPLET_NB = [3, 2, 2, 2, 1, 1, 1, 1]
SIZE_VALIDATION_SET = 1000

#########################################
#       FUNCTIONS                       #
#########################################


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


def define_datasets(with_test_set=False):
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
        db_list = ["cfp_humFiltered", "gbrieven_filtered", "lfw_filtered", "faceScrub_humanFiltered"]

        for i, db in enumerate(db_list):
            face_dic.update(pickle.load(open(FOLDER_DB + "faceDic_" + db + ".pkl", "rb")))

        # ------------------------------
        # 2. Shuffle all people
        # ------------------------------
        l = list(face_dic.items())
        random.shuffle(l)
        face_dic = dict(l)

        # ---------------------------------------------------------------------------
        # 3. Define "incremental" dictionaries (with 200, 300, 500... pictures in)
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
        # 4. Build Face_DS
        # -------------------------------------------

        face_DS_list = []
        for k, dic in enumerate(dict_list):
            face_DS_list.append(Face_DS(faces_dic=dic, nb_triplet=TRIPLET_NB[k]))

        # ----- Save Dataloader --------
        torch.save(face_DS_list, FOLDER_DB + name_ds_train + ".pt")
        print("The list of datasets has been saved! \n")

        # ---------------------------------------------------------
        # 5. Take the rest of data to build the validation set
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


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":
    datasets = define_datasets(with_test_set=True)
    model_name = None
    db_name_train = [FOLDER_DB + "gbrieven.zip", FOLDER_DB + "cfp.zip", FOLDER_DB + "lfw.zip",
                     FOLDER_DB + "faceScrub.zip"]

    test_id = 1

    if test_id == 1:
        # ------------------------- Tests Architectures and Losses -------------------------
        # 1. VGG16 & Triplet Loss
        # 2. VGG16 & Cross Entropy
        # 3. Basic Arch & Triplet Loss
        # 4. Basic Arch & Cross Entropy
        # 5. Basic Arch & Cross Entropy
        # => Choose best ------------------------------------------------------------------
        sets_list = [datasets[0][3], datasets[1], datasets[2]]
        _, model_name = main_train(sets_list, db_name_train, name_model=model_name)

    if test_id == 2:
        # ---------------- With best Test Data Quantity -------------------------
        for i in range(len(PICTURES_NB)):
            print("======================== Training on set " + str(i) + "... ====================================== ")
            sets_list = [datasets[0][i], datasets[1], datasets[2]]
            _, model_name = main_train(sets_list, db_name_train, name_model=model_name)
