import time
import torch.utils.data
import zipfile
import os
import pickle
import platform
import random
from random import shuffle
from string import digits
import torch.utils.data

from PIL import Image
from io import BytesIO

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import warnings

warnings.filterwarnings('ignore')

#########################################
#       GLOBAL VARIABLES                #
#########################################

if platform.system() == "Darwin":
    # if the 2th db not used, replace "yalefaces" by ""
    DB_TO_USE = ["Iranian"] #, "GTdbCrop", "yalefaces", "faces94", "Iranian", "painCrops", "utrecht"]
    MAIN_ZIP = 'datasets/ds0123456.zip'  # cfp.zip  # "testdb.zip"  CASIA-WebFace.zip' #
else:
    # if the 2th db not used, replace "yalefaces" by ""
    DB_TO_USE = ["AberdeenCrop", "GTdbCrop", "yalefaces", "faces94", "Iranian", "painCrops", "utrecht"]
    MAIN_ZIP = "/data/gbrieven/gbrieven.zip"  # "/data/gbrieven/CASIA-WebFace.zip"

if MAIN_ZIP.split("/")[-1] == 'CASIA-WebFace.zip':
    DB_TO_USE = None

ZIP_TO_PROCESS = 'datasets/cfp.zip'  # aber&GTdb_crop.zip'
NB_DIGIT_IN_ID = 1

SEED = 4  # When data is shuffled
EXTENSION = ".jpg"
SEPARATOR = "_"  # !!! Format of the dabase: name[!]_id !!!
RATION_TRAIN_SET = 1  # TO RESET !! 0.75
MAX_NB_ENTRY = 500000
MIN_NB_IM_PER_PERSON = 2
MAX_NB_IM_PER_PERSON = 20
MIN_NB_PICT_CLASSIF = 4  # s.t. 25% is used for testing
NB_TRIPLET_PER_PICT = 5

RATIO_HORIZ_CROP = 0.15
RESOLUTION = (150, 200)
CENTER_CROP = (150, 200)
TRANS = transforms.Compose([transforms.CenterCrop(CENTER_CROP), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])  # TO REMOVE, just for test


# ================================================================
#                    CLASS: Data
# ================================================================


class Data:
    def __init__(self, file, fn, db_path, to_process=False):
        self.file = file
        self.filename = fn  # Picture of the person
        self.db_path = db_path
        fn = fn.replace("/", SEPARATOR)
        self.db_source = fn.split(SEPARATOR)[0]

        if to_process:
            name_person, index, extension = self.extract_info_from_name()
        else:
            name_person = fn.split(SEPARATOR)[0] + fn.split(SEPARATOR)[1]  # = dbNamePersonName
            index = fn.split(SEPARATOR)[2]

            extension = fn.split(".")[1]
            self.db = fn.split(SEPARATOR)[0]

        self.name_person = name_person
        self.lateral_face = True if name_person[-1] == "!" else False
        self.index = index
        self.extension = extension

        # + Potentially add some characteristic related to the picture for the interpretation later, like girl/guy ....)

    '''---------------- convert_image --------------------------------
     This function converts the image into a jpeg image and extract it
     so that it can be included into the main "zip db"
     ---------------------------------------------------------------'''

    def convert_image(self):
        db_source = self.db_path.split("/")[1].split(".zip")[0]
        new_name = db_source + "_" + self.name_person + "_" + self.index + ".jpeg"

        if self.extension == "jpeg":
            # Rename under the formalism dbSource_personName_index.jpeg
            self.file.filename = new_name
            zipfile.ZipFile(self.db_path, 'r').extract(self.file)

        # Ok in the case of .gif at least
        elif self.extension != ".txt":
            zipfile.ZipFile(self.db_path, 'r').extract(self.file)
            Image.open(self.filename).convert('RGB').save(new_name)
            self.file.filename = new_name
            os.remove(self.filename)

    '''---------------- extract_info_from_name --------------------------------
     This function extracts from the name of the file: the name of the person, 
     the index of each picture and the extension of the file 
     --------------------------------------------------------------------------'''

    def extract_info_from_name(self):

        # === CASE 1: Right format is assumed ===

        if 1 < len(self.filename.split(SEPARATOR)) and self.filename.split(SEPARATOR)[1].split(".")[0].isdigit():
            # From the name of the image: namePerson_index.extension
            name_person = self.filename.split(SEPARATOR)[0]
            index = self.filename.split(SEPARATOR)[1].split(".")[0]
            extension = self.filename.split(".")[1]
            return name_person, index, extension

        # === CASE 2: No separation between the name of the person and the index ===
        number_if_error = 1

        # Check if there's no separator _ inside the name (if so delete it) f4001.jpg
        label = self.filename.translate(str.maketrans('', '', digits)).split(".")[0]
        extension = ".jpg"  # First default value

        try:
            digits_in_name = list(filter(str.isdigit, self.filename))
            label = label.replace("_", "") + "".join(digits_in_name[:NB_DIGIT_IN_ID])

            extension = "." + self.filename.split(".")[1]
            id = "".join(digits_in_name[len(digits_in_name) - NB_DIGIT_IN_ID:])

        except ValueError:
            print("There's no number id in the filename: " + str(self.filename))
            print("A number has been added then!")
            id = str(number_if_error)
            number_if_error += 1

        return label, id, extension

    '''---------------- resize_image --------------------------------
     This function resizes the image to RESOLUTION and first crops
     the image if the width is larger that heights by deleting 
     RATIO_HORIZ_CROP of the horizontal part on the left and on the 
     right. 
     OUT: the image data which has been processed 
     ---------------------------------------------------------------'''

    def resize_image(self):

        with zipfile.ZipFile(MAIN_ZIP, 'r') as archive:
            # Get image resolution
            original_image = Image.open(BytesIO(archive.read(self.filename))).convert("RGB")
            original_res = original_image.size
            # print("original res is: " + str(original_res))

            # ----- If Horizontal Image => Crop -----
            if original_res[1] < original_res[0]:
                left = RATIO_HORIZ_CROP * original_res[0]
                right = (1 - RATIO_HORIZ_CROP) * original_res[0]
                lower = original_res[1]
                upper = 0
                original_image = original_image.crop(box=(left, upper, right, lower))

            # ----- Set the resolution -----
            resized_image = original_image.resize(RESOLUTION)
            # plt.imshow(resized_image)
            # plt.show()

            return resized_image


# ================================================================
#                    CLASS: Fileset
# ================================================================


class Fileset:
    def __init__(self):
        self.data_list = []

    '''---------------- add_data -----------------------------------
     This function add data to the data_list 
     ---------------------------------------------------------------'''

    def add_data(self, data):
        self.data_list.append(data)

    '''---------------------- get_train_and_test_sets --------------------------------
     This function defines a training and a testing set from data_list by splitting it
     IN: diff_faces: True if no one has to be present in both training and testing sets
     OUT: a training and a testing sets that are Fileset objects 

     REM: no instantiation of the db_source variable ... 
     --------------------------------------------------------------------------------'''

    def get_train_and_test_sets(self, diff_faces, db_train=None, classification=False):
        print("Training and Testing Sets Definition ... \n")

        training_set = Fileset()
        testing_set = Fileset()

        # --------------------------------------------------------
        # CASE 1: same people in the training and the testing set
        # AND same number of picture/person in the training set
        # --------------------------------------------------------
        if classification:
            nb_pict_train = int(round(RATION_TRAIN_SET * MIN_NB_PICT_CLASSIF))
            curr_person = "none"  # !! Relies on the fact that pictures are ordered
            curr_pictures = []
            for i, data in enumerate(self.data_list):
                if curr_person == data.name_person:
                    curr_pictures.append(data)
                else:
                    # -------- Add to the training and testing sets the data related to the current person -------
                    if MIN_NB_PICT_CLASSIF <= len(curr_pictures):
                        j = 0
                        while j < nb_pict_train:
                            training_set.data_list.append(curr_pictures.pop())
                            j += 1
                        testing_set.data_list.extend(curr_pictures)
                    # ------- Reset Setting ------------
                    curr_person = data.name_person
                    curr_pictures = [data]

        # -------------------------------------------------------
        # CASE 2: same people in the training and the testing set
        # -------------------------------------------------------
        elif not diff_faces and db_train is None:
            random.Random(SEED).shuffle(self.data_list)
            training_set.data_list = self.data_list[:int(round(RATION_TRAIN_SET * len(self.data_list)))]
            testing_set.data_list = self.data_list[int(round(RATION_TRAIN_SET * len(self.data_list))):]

        # -------------------------------------------------------------
        # CASE 3: different people in the training and the testing set
        # -------------------------------------------------------------
        elif db_train is None:
            random.Random(SEED).shuffle(self.data_list)
            # Get all the names of the people in the ds
            all_labels = set()
            for i, data in enumerate(self.data_list):
                all_labels.add(data.name_person)

            # Build a training and a testing set with different labels
            for i in range(int(round((1 - RATION_TRAIN_SET) * len(all_labels)))): all_labels.pop()

            for i, data in enumerate(self.data_list):
                if data.name_person in all_labels:
                    training_set.data_list.append(data)
                else:
                    testing_set.data_list.append(data)

        # -------------------------------------------------------
        # CASE 4: Specific DB for training and testing sets
        # -------------------------------------------------------
        else:
            for i, data in enumerate(self.data_list):
                if data.db_path in db_train:
                    training_set.data_list.append(data)
                else:
                    testing_set.data_list.append(data)

        return training_set, testing_set

    '''---------------- order_per_personName --------------------------------
     This function returns a dictionary where the key is the name of the 
     person and the value is a list of images of this person 

     IN: transform: transformation that has to be applied to the picture 
     OUT: dictionary where the key is the name of the person and the value is 
     the list of their pictures as FaceImage object
     ----------------------------------------------------------------------- '''

    def order_per_personName(self, transform, nb_people=None, max_nb_pict=MAX_NB_IM_PER_PERSON,
                             min_nb_pict=MIN_NB_IM_PER_PERSON, same_nb_pict=False):

        faces_dic = {}
        random.Random(SEED).shuffle(self.data_list)

        # --------------------------------
        # Order the picture per label
        # --------------------------------
        for i, data in enumerate(self.data_list):
            personNames = data.name_person
            res_image = data.resize_image()
            formatted_image = transform(res_image)
            img = FaceImage(data.filename, formatted_image)

            try:
                if len(faces_dic[personNames]) < max_nb_pict:
                    faces_dic[personNames].append(img)
            except KeyError:
                if nb_people is None or len(faces_dic) < nb_people:
                    faces_dic[personNames] = [img]

        if same_nb_pict:
            pictures_nbs = [len(pictures_list) for person_name, pictures_list in faces_dic.items()]
            min_nb_pict = min(pictures_nbs)

        # --- Remove element where value doesn't contain enough pictures -----
        faces_dic = {label: pictures for label, pictures in faces_dic.items() if min_nb_pict <= len(pictures)}

        return faces_dic

    '''---------------------------- ds_to_zip --------------------------------
     This function adds to zip_filename all the content of data_list, such that
     each image is in the jpeg format and the name of each file has the right
     format (i.e. dbSource_personName_id.jpg) 
     --------------------------------------------------------------------------'''

    def ds_to_zip(self):
        zipf = zipfile.ZipFile(MAIN_ZIP, 'a', zipfile.ZIP_DEFLATED)

        for i, data in enumerate(self.data_list):
            # Convert into jpeg and extract the image
            data.convert_image()

            zipf.write(data.file.filename, os.path.basename(data.file.filename))
            os.remove(data.file.filename)

        zipf.close()


# ================================================================
#                    CLASS: FaceImage
# ================================================================
class FaceImage():
    def __init__(self, path, trans_image):
        self.path = path
        self.trans_img = trans_image
        self.feature_repres = None

    def isIqual(self, other_image):
        return other_image.path == self.path

    def display_im(self, to_print="A face is displayed"):
        print(to_print)
        with zipfile.ZipFile(MAIN_ZIP, 'r') as archive:
            image = Image.open(BytesIO(archive.read(self.path))).convert("RGB")
            plt.imshow(image)
            plt.show()

    def get_feature_repres(self, model):
        if self.feature_repres is not None:
            return self.feature_repres
        else:
            data = torch.unsqueeze(self.trans_img, 0)
            self.feature_repres = model.embedding_net(data)
            return self.feature_repres


# ================================================================
#                    CLASS: Face_DS
# ================================================================


class Face_DS(torch.utils.data.Dataset):
    def __init__(self, fileset, transform=TRANS, to_print=False, device="cpu", triplet_version=True, save=None):

        self.to_print = to_print
        self.transform = transforms.ToTensor() if transform is None else transform

        faces_dic = fileset.order_per_personName(self.transform)

        # print("after transformation" + str(faces_dic['faces94jdbenm'][0]))

        self.train_data = []
        self.train_labels = []
        self.nb_classes = 0

        # ------ Build triplet supporting the dataset (ensures balanced classes) --------
        if triplet_version:
            self.train_not_formatted_data = []
            self.build_triplet(faces_dic, device=device)

        # -------- Build training set composed of faces --------
        else:
            self.image_data(faces_dic, device=device)

        self.print_data_report(faces_dic)
        if save is not None:
            db = MAIN_ZIP.split("/")[-1].split(".")[0]
            pickle.dump(self, open("datasets/" + save + db + ".pkl", "wb"), protocol=2)
            print("The set has been saved!\n")

    # You must override __getitem__ and __len__
    def __getitem__(self, index, visualization=False):
        """ ---------------------------------------------------------------------------------------------
            An item is made up of 3 images (P, P, N) and 2 target (1, 0) specifying that the 2 first
            images are the same and the first and the third are different. The images are represented
            through TENSORS.

            If visualize = True: the image is printed
        ----------------------------------------------------------------------------------------------- """
        if self.to_print:
            visualization = True
            print("IN GET ITEM: the index in the dataset is: " + str(index) + "\n")

        if visualization:
            for i, image_face in enumerate(self.train_not_formatted_data[index]):
                print("Face " + str(i) + ": ")
                image_face.display_im()

        return self.train_data[index], self.train_labels[index]

    def __len__(self):
        """ -------------------------------------------
        Total number of samples in the dataset
        ----------------------------------------------- """
        return len(self.train_data)

    def build_triplet(self, faces_dic, device="cpu"):
        """ ---------------------------------------------------
        Define the training set composed of (ref, pos, neg)
        List of 3 lists, each containing tensors

        !! Convention !!:
            Label = 0 if same people
            Label = 1 if different people
        --------------------------------------------------- """
        print("In triplet building...")
        print("face dic is " + str(faces_dic))
        all_labels = list(faces_dic.keys())
        nb_labels = len(all_labels)

        # ================= Consider each person =================
        for label, pictures_list in faces_dic.items():
            labels_indexes_neg = [x for x in range(0, nb_labels) if x != all_labels.index(label)]
            pos_pict_lists = []

            # ================= Consider each picture of the person =================
            for i, picture_ref in enumerate(pictures_list):
                pos_pict_list = []
                pic_ind_pos = list(range(len(pictures_list)))

                # ================= Consider several times the ref picture =================
                for j in range(NB_TRIPLET_PER_PICT):  # !! TO CHECK with very small db
                    try:
                        curr_index_pos = random.choice(pic_ind_pos)

                        while (picture_ref.isIqual(pictures_list[curr_index_pos]) or
                                   (curr_index_pos < i and i in pos_pict_lists[curr_index_pos])):
                            pic_ind_pos.remove(curr_index_pos)
                            curr_index_pos = random.choice(pic_ind_pos)

                    except IndexError:
                        # !! Means that the current label has no remaining other picture
                        # (empty sequence of available index)
                        break

                    picture_positive = pictures_list[curr_index_pos]
                    pic_ind_pos.remove(curr_index_pos)
                    pos_pict_list.append(curr_index_pos)  # Remember the pairs that are defined to avoid redundancy

                    # Pick a random different person
                    label_neg = all_labels[random.choice(labels_indexes_neg)]
                    picture_negative = random.choice(faces_dic[label_neg])

                    # To keep track of the image itself in order to potentially print it
                    self.train_not_formatted_data.append([picture_ref, picture_positive, picture_negative])

                    self.train_data.append([picture_ref.trans_img.to(device), picture_positive.trans_img.to(device),
                                            picture_negative.trans_img.to(device)])

                    self.train_labels.append([0, 1])

                pos_pict_lists.append(pos_pict_list)

        # self.train_data = torch.stack(self.train_data)
        self.train_labels = torch.tensor(self.train_labels).to(device)

    """ --------------------- image_data ------------------------------------
      This function set data by filling it with the pictures contained in 
      face_dic (that are the elements in the values)
      --------------------------------------------------------------------- """

    def image_data(self, faces_dic, device="cpu"):

        label_nb = 0
        # ========= Consider each person =================
        for label, pictures_list in faces_dic.items():
            # ======== Consider each picture of each person ==========
            for i, picture in enumerate(pictures_list):
                self.train_data.append(picture.trans_img.to(device))
                self.train_labels.append(label_nb)
            label_nb += 1
        self.nb_classes = len(faces_dic)
        self.train_labels = torch.tensor(self.train_labels).to(device)

    """ ---------------------------------- print_data_report ------------------------------------  """

    def print_data_report(self, faces_dic):

        # Report about the quantity of data
        pictures_nbs = [len(pictures_list) for person_name, pictures_list in faces_dic.items()]
        max_nb_pictures = max(pictures_nbs)
        min_nb_pictures = min(pictures_nbs)

        print("\n ---------------- Report about the quantity of data  -------------------- ")
        print("The total quantity of pairs used as data is: " + str(2 * len(self.train_labels)))
        print("The number of different people in set is: " + str(len(list(faces_dic.keys()))))
        print("The number of pictures per person is between: " + str(min_nb_pictures) + " and " + str(max_nb_pictures))
        print("The average number of pictures per person is: " + str(sum(pictures_nbs) / len(pictures_nbs)))
        print(" ------------------------------------------------------------------------\n")


# ================================================================
#                    FUNCTIONS
# ================================================================


'''---------------- load_sets -------------------------------- 
This function load the training and the testing sets derived
from the specified db, if there's any 
-------------------------------------------------------------- '''


def load_sets():
    # Try to Load train and test sets that were already defined
    db = MAIN_ZIP.split("/")[-1].split(".")[0]

    with open("datasets/trainset_" + db + ".pkl", "rb") as f:
        training_set = pickle.load(f)
    with open("datasets/testset_" + db + ".pkl", "rb") as f:
        testing_set = pickle.load(f)

    return training_set, testing_set


'''--------------------- include_data_to_zip --------------------------------
 This function adds to MAIN_ZIP the processed content of ZIP_TO_PROCESS
 --------------------------------------------------------------------------'''


def include_data_to_zip():
    dataset = Fileset()

    with zipfile.ZipFile(ZIP_TO_PROCESS, 'r') as archive:
        file_names = archive.namelist()
        file_list = archive.filelist

        # Create data from each image and add it to the dataset
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
            if fn == ".DS_Store" or fn == "__MACOSX/" or fn[-1] == "/":
                continue

            new_data = Data(file_list[i], fn, ZIP_TO_PROCESS, True)
            dataset.add_data(new_data)

    dataset.ds_to_zip()


'''----------------------------- from_zip_to_data --------------------------------------------
 This function turns the content of the MAIN ZIP into a Fileset object
 !! 2 possible structures inside the main zip: 
 1. set of files whose name is like dbOrigin_person_id.jpg
 2. a folder containing folders (related to each person) containing files whose name is id.jpg
 ---------------------------------------------------------------------------------------------'''


def from_zip_to_data(with_profile, fname=MAIN_ZIP):
    t = time.time()
    dataset = Fileset()
    if fname is None: fname = MAIN_ZIP

    print("\nData Loading ...")
    with zipfile.ZipFile(fname, 'r') as archive:
        file_names = archive.namelist()
        file_list = archive.filelist

        nb_entry = 0
        nb_entry_pers = 0
        previous_person = ""
        go_to_next_folder = False

        # ------ Create data from each image and add it to the dataset ----
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
            if fn[-1] == "/":
                go_to_next_folder = False
                continue
            if go_to_next_folder:
                continue

            # --- Just keep a limited among of pictures per person ---
            try:
                if previous_person == fn.split("/")[1]:
                    if MAX_NB_IM_PER_PERSON < nb_entry_pers:
                        nb_entry_pers = 1
                        go_to_next_folder = True
                        continue
                    else:
                        nb_entry_pers += 1
                else:
                    nb_entry_pers = 1
                    previous_person = fn.split("/")[1]
            except IndexError:  # case where the structure of the db isn't through folders
                pass

            new_data = Data(file_list[i], fn, MAIN_ZIP, False)

            if (with_profile or not new_data.lateral_face) and (DB_TO_USE is None or new_data.db in DB_TO_USE):
                dataset.add_data(new_data)

            # ------ Limitation of the total number of instances ------
            nb_entry += 1
            if MAX_NB_ENTRY < nb_entry:
                break

    print("Loading Time: " + str(time.time() - t))
    print(str(len(dataset.data_list)) + " pictures have been loaded!\n")
    return dataset


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":
    fileset = from_zip_to_data(False)
    training_set, _ = fileset.get_train_and_test_sets(False)
    ts = Face_DS(training_set)  # Triplet Version
