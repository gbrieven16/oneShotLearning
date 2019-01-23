import zipfile
import os
import random
from random import shuffle
from string import digits
import torch.utils.data

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.utils.data

import warnings

warnings.filterwarnings('ignore')

#########################################
#       GLOBAL VARIABLES                #
#########################################

DB_TO_USE = ["AberdeenCrop", "GTdbCrop", "yalefaces", "faces94"] #, "Iranian"]  # if the 2th db not used, replace "yalefaces" by ""
MAIN_ZIP = 'datasets/ds01234.zip'

ZIP_TO_PROCESS = 'datasets/Iranian.zip'  # aber&GTdb_crop.zip'
NB_DIGIT_IN_ID = 2

SEED = 4  # When data is shuffled
EXTENSION = ".jpg"
SEPARATOR = "_"  # !!! Format of the dabase: name[!]_id !!!
RATION_TRAIN_SET = 0.75
MAX_NB_IM_PER_PERSON = 100


# ================================================================
#                    CLASS: Data
# ================================================================


class Data:
    def __init__(self, file, fn, db_path, to_process=False):
        self.file = file
        self.filename = fn  # Picture of the person
        self.db_path = db_path

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

        # Check if there's no separator _ inside the name (if so delete it)
        label = self.filename.translate(str.maketrans('', '', digits)).split(".")[0]

        try:
            digits_in_name = list(filter(str.isdigit, self.filename))
            label = label.replace("_", "") + "".join(digits_in_name[:NB_DIGIT_IN_ID])

            extension = "." + self.filename.split(".")[1]
            id = "".join(digits_in_name[len(digits_in_name) - NB_DIGIT_IN_ID:])

            if id[1] in ["3", "4", "5", "6", "9", "0"]:
                label = label + "!"

        except ValueError:
            print("There's no number id in the filename: " + str(self.filename))
            print("A number has been added then!")
            id = str(number_if_error)
            number_if_error += 1

        return label, id, extension


# ================================================================
#                    CLASS: Fileset
# ================================================================


class Fileset:
    def __init__(self):
        self.db_source = []
        self.data_list = []

    '''---------------- add_data -----------------------------------
     This function add data to the data_list 
     ---------------------------------------------------------------'''

    def add_data(self, data):
        if data.db_path not in self.db_source:
            self.db_source.append(data.db_path)

        self.data_list.append(data)

    '''---------------------- get_train_and_test_sets --------------------------------
     This function defines a training and a testing set from data_list by splitting it
     IN: diff_faces: True if no one has to be present in both training and testing sets
     OUT: a training and a testing sets that are Fileset objects 
     
     REM: no instantiation of the db_source variable ... 
     --------------------------------------------------------------------------------'''

    def get_train_and_test_sets(self, diff_faces):

        random.Random(SEED).shuffle(self.data_list)
        training_set = Fileset()
        testing_set = Fileset()

        ############################################################
        # CASE 1: same people in the training and the testing set
        ############################################################
        if not diff_faces:
            training_set.data_list = self.data_list[:round(RATION_TRAIN_SET * len(self.data_list))]
            testing_set.data_list = self.data_list[round(RATION_TRAIN_SET * len(self.data_list)):]

        ##############################################################
        # CASE 2: different people in the training and the testing set
        ##############################################################
        else:
            # Get all the names of the people in the ds
            all_labels = set()
            for i, data in enumerate(self.data_list):
                all_labels.add(data.name_person)

            # Build a training and a testing set with different labels
            for i in range(round((1 - RATION_TRAIN_SET) * len(all_labels))): all_labels.pop()

            for i, data in enumerate(self.data_list):
                if data.name_person in all_labels:
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

    def order_per_personName(self, transform, nb_people=None, max_nb_pictures=MAX_NB_IM_PER_PERSON):

        faces_dic = {}
        random.Random(SEED).shuffle(self.data_list)

        #################################
        # Order the picture per label
        #################################
        with zipfile.ZipFile(MAIN_ZIP, 'r') as archive:
            for i, data in enumerate(self.data_list):
                personNames = data.name_person

                formatted_image = transform(Image.open(BytesIO(archive.read(data.filename))).convert("RGB"))
                img = FaceImage(data.filename, formatted_image)

                try:
                    if len(faces_dic[personNames]) < max_nb_pictures:
                        faces_dic[personNames].append(img)
                except KeyError:
                    if nb_people is None or len(faces_dic) < nb_people:
                        faces_dic[personNames] = [img]
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

    def isIqual(self, other_image):
        return other_image.path == self.path

    def display_im(self, to_print="A face is displayed"):
        print(to_print)
        with zipfile.ZipFile(MAIN_ZIP, 'r') as archive:
            image = Image.open(BytesIO(archive.read(self.path))).convert("RGB")
            plt.imshow(image)
            plt.show()


# ================================================================
#                    CLASS: Face_DS
# ================================================================


class Face_DS(torch.utils.data.Dataset):
    def __init__(self, fileset, transform=None, to_print=False, device="cpu"):

        self.to_print = to_print
        self.transform = transforms.ToTensor() if transform is None else transform

        faces_dic = fileset.order_per_personName(self.transform)
        # print("after transformation" + str(faces_dic['faces94jdbenm'][0]))

        ########################################################################
        # Build triplet supporting the dataset (ensures balanced classes)
        ########################################################################
        self.train_data = []
        self.train_labels = []
        self.train_not_formatted_data = []
        self.build_triplet(faces_dic, device=device)

        self.print_data_report(faces_dic)

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

        all_labels = list(faces_dic.keys())
        nb_labels = len(all_labels)

        # ================= Consider each person =================
        for label, pictures_list in faces_dic.items():
            pic_ind_pos = list(range(len(pictures_list)))
            shuffle(pic_ind_pos)
            labels_indexes_neg = [x for x in range(0, nb_labels) if x != all_labels.index(label)]

            # ================= Consider each picture of the person =================
            for i, picture_ref in enumerate(pictures_list):

                try:
                    if not picture_ref.isIqual(pictures_list[pic_ind_pos[-1]]):
                        picture_positive = pictures_list[pic_ind_pos.pop()]
                    else:
                        picture_positive = pictures_list[pic_ind_pos.pop(-2)]
                except IndexError:
                    # !! Means that the current label has no remaining other picture !!
                    break

                # Pick a random different person
                label_neg = all_labels[random.choice(labels_indexes_neg)]
                picture_negative = random.choice(faces_dic[label_neg])

                # To keep track of the image itself in order to potentially print it
                self.train_not_formatted_data.append([picture_ref, picture_positive, picture_negative])

                self.train_data.append([picture_ref.trans_img.to(device), picture_positive.trans_img.to(device),
                                            picture_negative.trans_img.to(device)])

                self.train_labels.append([0, 1])

        # self.train_data = torch.stack(self.train_data)
        self.train_labels = torch.tensor(self.train_labels).to(device)

    """ ---------------------------------- print_data_report ------------------------------------  """

    def print_data_report(self, faces_dic):

        # Report about the quantity of data
        pictures_nbs = [len(pictures_list) for person_name, pictures_list in faces_dic.items()]
        max_nb_pictures = max(pictures_nbs)
        min_nb_pictures = 1

        print("\n ---------------- Report about the quantity of data  -------------------- ")
        print("The total quantity of pairs used as data is: " + str(2 * len(self.train_labels)))
        print("The number of different people in set is: " + str(len(list(faces_dic.keys()))))
        print("The number of pictures per person is between: " + str(min_nb_pictures) + " and " + str(max_nb_pictures))
        print("The average number of pictures per person is: " + str(sum(pictures_nbs) / len(pictures_nbs)))
        print(" ------------------------------------------------------------------------\n")


'''--------------------- from_zip_to_data --------------------------------
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


'''--------------------- from_zip_to_data --------------------------------
 This function turns the content of the MAIN ZIP into a Fileset object
 --------------------------------------------------------------------------'''


def from_zip_to_data(with_profile):
    dataset = Fileset()

    with zipfile.ZipFile(MAIN_ZIP, 'r') as archive:
        file_names = archive.namelist()
        file_list = archive.filelist

        # Create data from each image and add it to the dataset
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension

            new_data = Data(file_list[i], fn, MAIN_ZIP, False)

            if (with_profile or not new_data.lateral_face) and new_data.db in DB_TO_USE:
                dataset.add_data(new_data)
    return dataset


if __name__ == "__main__":
    include_data_to_zip()