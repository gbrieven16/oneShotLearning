import os
import time
import pickle
import platform
import zipfile
import random
from string import digits
from StyleEncoder import data_augmentation
import torch.utils.data

from PIL import Image
from io import BytesIO

import matplotlib

matplotlib.use("TkAgg")  # ('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch import nn
from Face_alignment import align_faces, ALIGNED_IMAGES_DIR

import warnings

warnings.filterwarnings('ignore')

#########################################
#       GLOBAL VARIABLES                #
#########################################


# if the 2th db not used, replace "yalefaces" by ""
FOLDER_DB = "data/gbrieven/" if platform.system() == "Darwin" else "/data/gbrieven/"
MAIN_ZIP = FOLDER_DB + 'cfp.zip'  # cfp.zip "testdb.zip"  CASIA-WebFace.zip'
TEST_ZIP = FOLDER_DB + 'testdb.zip'

ZIP_TO_PROCESS = FOLDER_DB + 'initTestCropped.zip'  # aber&GTdb_crop.zip'
NB_DIGIT_IN_ID = 1

SEED = 4  # When data is shuffled
EXTENSION = ".jpg"
SEPARATOR = "__"  # !!! Format of the dabase: name[!]__id !!! & No separators in name!
RATION_TRAIN_SET = 0.75
MAX_NB_ENTRY = 500000
MIN_NB_IM_PER_PERSON = 2
MAX_NB_IM_PER_PERSON = 20
MIN_NB_PICT_CLASSIF = 4  # s.t. 25% is used for testing
NB_TRIPLET_PER_PICT = 2
WITH_AL = False

RATIO_HORIZ_CROP = 0.15
CENTER_CROP = (200, 150)
TRANS = transforms.Compose([transforms.CenterCrop(CENTER_CROP), transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])

Q_DATA_AUGM = 8
DIST_METRIC = "Manhattan"  # "Cosine_Sym"


# ================================================================
#                    CLASS: Data
# ================================================================


class Data:
    def __init__(self, fn, db_path, to_process=False, picture=None):
        self.filename = fn  # Picture of the person
        self.db_path = db_path
        fn = fn.replace("/", SEPARATOR)
        self.db_source = fn.split(SEPARATOR)[0]

        if picture is not None:
            self.file = picture

        if to_process:
            # name_person, index, extension = self.extract_info_from_name()
            name_person = fn.split(SEPARATOR)[1]  # = dbNamePersonName
            index = fn.split(SEPARATOR)[2].split(".")[0]

            extension = ".".join(fn.split(".")[1:])
        else:
            name_person = fn.split(SEPARATOR)[0] + fn.split(SEPARATOR)[1]  # = dbNamePersonName
            index = fn.split(SEPARATOR)[2].split(".")[0]

            extension = ".".join(fn.split(".")[1:])

        self.name_person = name_person
        self.lateral_face = True if name_person[-1] == "!" else False
        self.synthetic = True if len(index.split("_")) == 2 else False
        self.index = index
        self.extension = extension

        '''print("db_source is " + self.db_source)
        print("person is " + self.name_person)
        print("index " + str(self.index))
        print("extension " + str(extension))'''

        # + Potentially add some characteristic related to the picture for the interpretation later, like girl/guy ....)

    '''---------------- convert_image --------------------------------
     This function converts the image into a jpeg image and extract it
     so that it can be included into the main "zip db"
     ---------------------------------------------------------------'''

    def convert_image(self, file_is_pil=False):

        new_name = self.db_source + SEPARATOR + self.name_person + SEPARATOR + self.index + ".jpg"

        if file_is_pil:
            self.file.save(FOLDER_DB + new_name)
            return FOLDER_DB + new_name, new_name

        elif self.extension == "jpg":
            # Rename under the formalism dbSource_personName_index.jpeg
            self.file.filename = new_name
            zipfile.ZipFile(self.db_path, 'r').extract(self.filename)

        # Ok in the case of .gif at least
        elif self.extension != ".txt" and self.extension != "png.json":

            with zipfile.ZipFile(self.db_path, 'r') as archive:
                path = FOLDER_DB + "image.jpg"
                Image.open(BytesIO(archive.read(self.filename))).convert('RGB').save(path)
                self.file.filename = new_name

            return path, new_name



# ================================================================
#                    CLASS: Fileset
# ================================================================


class Fileset:
    def __init__(self):
        self.data_list = []
        self.all_db = []

    '''---------------- add_data -----------------------------------
     This function add data to the data_list 
     ---------------------------------------------------------------'''

    def add_data(self, data):
        self.data_list.append(data)
        if data.db_source not in self.all_db:
            self.all_db.append(data.db_source)

    '''--------------------------- get_sets -------------------------------------------
     This function defines 2 sets (basically a training and a validation one) 
     from data_list by splitting it
     IN: diff_faces: True if no one has to be present in both training and testing sets
         db_set1: list of the db providing the content of set1 (and the rest if for set2) 
     OUT: a training and a testing sets that are Fileset objects 

     REM: no instantiation of the db_source variable ... 
     --------------------------------------------------------------------------------'''

    def get_sets(self, diff_faces, db_set1=None, classification=False, nb_classes=None):

        set1 = Fileset()
        set2 = Fileset()

        # --------------------------------------------------------
        # CASE 1: same people in the 2 sets AND same number of
        # picture/person in the first set (i.e training set)
        # --------------------------------------------------------
        if classification:
            nb_pict_train = int(round(RATION_TRAIN_SET * MIN_NB_PICT_CLASSIF))
            curr_person = "none"  # !! Relies on the fact that pictures are ordered
            curr_pictures = []
            nb_people = 0
            for i, data in enumerate(self.data_list):
                if curr_person == data.name_person:
                    curr_pictures.append(data)
                else:
                    # --- Increment and check the total number of different people that is considered ---
                    nb_people += 1
                    if nb_classes < nb_people:
                        break
                    # -------- Add to the training and valid. sets the data related to the current person -------
                    if MIN_NB_PICT_CLASSIF <= len(curr_pictures):
                        j = 0
                        # Take nb_pict_train from the list of the current pictures to put in the training set
                        while j < nb_pict_train:
                            set1.add_data(curr_pictures.pop())
                            j += 1
                        # Empty the rest of the list of the current pictures in the validation set
                        for i, picture in enumerate(curr_pictures):
                            set2.add_data(picture)
                    # ------- Reset Setting ------------
                    curr_person = data.name_person
                    curr_pictures = [data]

            print("The total number of people considered in the classification is: " + str(nb_people))

        # -------------------------------------------------------
        # CASE 2: same people in the training and the testing set
        # -------------------------------------------------------
        elif not diff_faces and db_set1 is None:
            print("Definition of 2 sets from the same database, with same people ...")
            random.Random(SEED).shuffle(self.data_list)
            set1.all_db = self.all_db
            set2.all_db = self.all_db
            set1.data_list = self.data_list[:int(round(RATION_TRAIN_SET * len(self.data_list)))]
            set2.data_list = self.data_list[int(round(RATION_TRAIN_SET * len(self.data_list))):]

        # -------------------------------------------------------------
        # CASE 3: different people in 2 sets
        # -------------------------------------------------------------
        elif db_set1 is None:
            print("Definition of 2 sets from the same database, with different people ...")
            random.Random(SEED).shuffle(self.data_list)
            # Get all the names of the people in the ds
            all_labels = set()
            for i, data in enumerate(self.data_list):
                all_labels.add(data.name_person)

            # Build a training and a testing set with different labels
            for i in range(int(round((1 - RATION_TRAIN_SET) * len(all_labels)))): all_labels.pop()

            for i, data in enumerate(self.data_list):
                if data.name_person in all_labels:
                    set1.add_data(data)
                else:
                    set2.add_data(data)

        # -------------------------------------------------------
        # CASE 4: Specific DB for training and testing sets
        # -------------------------------------------------------
        else:
            print("Definition of 1 set related to database: " + str(db_set1))
            for i, data in enumerate(self.data_list):
                if data.db_source in db_set1:
                    set1.add_data(data)
                else:
                    print("Definition of a second set related to database: " + str(data.db_source))
                    set2.add_data(data)

        return set1, set2

    '''---------------- order_per_personName --------------------------------
     This function returns a dictionary where the key is the name of the 
     person and the value is a list of images of this person 

     IN: transform: transformation that has to be applied to the picture 
     OUT: dictionary where the key is the name of the person and the value is 
     the list of their pictures as FaceImage object
     ----------------------------------------------------------------------- '''

    def order_per_personName(self, transform, nb_people=None, max_nb_pict=MAX_NB_IM_PER_PERSON,
                             min_nb_pict=MIN_NB_IM_PER_PERSON, save=None):

        faces_dic = {}
        random.Random(SEED).shuffle(self.data_list)
        nb_undetected_faces = 0

        # --------------------------------
        # Order the picture per label
        # --------------------------------
        for i, data in enumerate(self.data_list):
            personName = data.name_person

            with zipfile.ZipFile(data.db_path, 'r') as archive:
                original_image = Image.open(BytesIO(archive.read(data.filename))).convert("RGB")
            res_image = align_faces(original_image) if WITH_AL else original_image

            try:
                formatted_image = transform(res_image)
            except AttributeError:
                # print("No face was detected on: " + data.filename + " with index " + str(i))
                original_image.save("Undetected_" + data.filename.replace("/", SEPARATOR))
                nb_undetected_faces += 1
                continue

            img = FaceImage(data.filename, formatted_image, db_path=data.db_path, pers=personName, i=data.index)

            try:
                if len(faces_dic[personName]) < max_nb_pict:
                    faces_dic[personName].append(img)
            except KeyError:
                if nb_people is None or len(faces_dic) < nb_people:
                    faces_dic[personName] = [img]

        print("The total number of pictures where the face wasn't detected is: " + str(nb_undetected_faces))
        # --- Remove element where value doesn't contain enough pictures -----
        faces_dic = {label: pictures for label, pictures in faces_dic.items() if min_nb_pict <= len(pictures)}

        if save is not None:
            try:
                pickle.dump(faces_dic, open(FOLDER_DB + "faceDic_" + save + ".pkl", "wb"))
            except OSError:
                try:
                    mid = int(round(len(faces_dic) / 2))
                    faces_dic1 = {k: v for k, v in faces_dic.items() if k in list(faces_dic)[:mid]}
                    faces_dic2 = {k: v for k, v in faces_dic.items() if k in list(faces_dic)[mid:]}

                    pickle.dump(faces_dic1, open(FOLDER_DB + "faceDic_" + save + "1.pkl", "wb"))
                    pickle.dump(faces_dic2, open(FOLDER_DB + "faceDic_" + save + "2.pkl", "wb"))
                except OSError:
                    mid1 = int(round(len(faces_dic1) / 2))
                    faces_dic11 = {k: v for k, v in faces_dic1.items() if k in list(faces_dic1)[:mid1]}
                    faces_dic12 = {k: v for k, v in faces_dic1.items() if k in list(faces_dic1)[mid1:]}

                    pickle.dump(faces_dic11, open(FOLDER_DB + "faceDic_" + save + "1.pkl", "wb"))
                    pickle.dump(faces_dic12, open(FOLDER_DB + "faceDic_" + save + "2.pkl", "wb"))

                    faces_dic21 = {k: v for k, v in faces_dic2.items() if k in list(faces_dic2)[:mid1]}
                    faces_dic22 = {k: v for k, v in faces_dic2.items() if k in list(faces_dic2)[mid1:]}

                    pickle.dump(faces_dic21, open(FOLDER_DB + "faceDic_" + save + "3.pkl", "wb"))
                    pickle.dump(faces_dic22, open(FOLDER_DB + "faceDic_" + save + "4.pkl", "wb"))

        return faces_dic

    '''---------------------------- ds_to_zip --------------------------------------
     This function adds to zip_filename all the content of data_list, such that
     each image is in the jpeg format and the name of each file has the right
     format (i.e. dbSource_personName_id.jpg) 
     IN: db_destination: db where to put the images contained in data_list 
         file_is_pil: True if the file of the data in datalist are of the PIL.image
     -------------------------------------------------------------------------------'''

    def ds_to_zip(self, db_destination=None, file_is_pil=False):
        if db_destination is None:
            db_destination = MAIN_ZIP
        zipf = zipfile.ZipFile(db_destination, 'a', zipfile.ZIP_DEFLATED)
        curr_pers = ""
        curr_nb_pict = 0

        for i, data in enumerate(self.data_list):
            if curr_pers == data.name_person:
                if MAX_NB_IM_PER_PERSON < curr_nb_pict:
                    continue
                else:
                    curr_nb_pict += 1
            else:
                curr_pers = data.name_person
                curr_nb_pict = 1

            # Convert into jpeg and extract the image
            try:
                if file_is_pil:
                    path, new_name = data.convert_image(file_is_pil=file_is_pil)
                else:
                    path, new_name = data.convert_image()

                # zipf.write(data.file.filename)
                zipf.write(path, new_name)
                os.remove(path)
            except TypeError:  # ".png.json" extension
                pass
            except OSError:
                pass  # There's a "." in the name of the person ><

        zipf.close()


# ================================================================
#                    CLASS: FaceImage
# ================================================================
class FaceImage():
    def __init__(self, file_path, trans_image, db_path=None, pers=None, i=None):
        self.file_path = file_path  # Complete path of the image
        self.db_path = db_path  # Complete path of the db (zip file)
        self.trans_img = trans_image
        self.dist = {}  # key1 person, val1: dic2 ; key2: index, val2: val2
        self.feature_repres = None
        self.person = pers
        self.index = i

    def isIqual(self, other_image):
        # return other_image.path == self.file_path
        return other_image.file_path == self.file_path

    def display_im(self, to_print="A face is displayed"):
        print(to_print)
        with zipfile.ZipFile(self.db_path, 'r') as archive:
            image = Image.open(BytesIO(archive.read(self.file_path))).convert("RGB")

            plt.imshow(image)
            plt.show()
            image.close()

    def get_feature_repres(self, model):
        if self.feature_repres is not None:
            return self.feature_repres
        else:
            data = torch.unsqueeze(self.trans_img, 0)
            self.feature_repres = model.embedding_net(data)
            return self.feature_repres

    def get_dist(self, person, index, picture, siamese_model):
        try:
            return self.dist[person][index]
        except KeyError:
            feature_repr2 = picture.get_feature_repres(siamese_model)
            if DIST_METRIC == "Manhattan":
                difference_sum = torch.sum(torch.abs(feature_repr2 - self.feature_repres))
                dist = float(difference_sum / len(self.feature_repres[0]))

            elif DIST_METRIC == "MeanSquare":
                difference_sum = torch.sum((self.feature_repres - feature_repr2) ** 2)
                dist = float(difference_sum / len(self.feature_repres[0]))

            elif DIST_METRIC == "Cosine_Sym":
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                dist = float(cos(self.feature_repres, feature_repr2))
            else:
                print("ERR: Invalid Distance Metric")
                raise IOError

            try:
                picture.dist[self.person][self.index] = dist
            except KeyError:
                picture.dist[self.person] = {}
                picture.dist[self.person][self.index] = dist

            return dist


# ================================================================
#                    CLASS: Face_DS
# ================================================================


class Face_DS(torch.utils.data.Dataset):
    def __init__(self, fileset=None, face_set=None, transform=TRANS, to_print=False, device="cpu",
                 triplet_version=True, save=None):

        self.to_print = to_print
        self.transform = transforms.ToTensor() if transform is None else transform
        self.train_data = []
        self.train_labels = []
        self.nb_classes = 0

        # -------------------------------------------------------------------------------
        # CASE 1: Build a non-triplet version dataset from a triplet version dataset
        # -------------------------------------------------------------------------------

        if face_set is not None:
            for i, data in enumerate(face_set.train_data):
                self.train_data.append(data[0])
                self.train_labels.append(i)
            self.train_labels = torch.tensor(self.train_labels)
            return

        self.all_db = fileset.all_db

        t = time.time()
        faces_dic = fileset.order_per_personName(self.transform)
        print("Pictures have been processed and ordered after " + str(time.time() - t))

        # data_augmentation(face_dic, Q_DATA_AUGM)

        # ------------------------------------------------------------------------
        # CASE 2: Build triplet supporting the dataset (ensures balanced classes)
        # ------------------------------------------------------------------------
        if triplet_version:
            self.train_not_formatted_data = []
            self.build_triplet(faces_dic, device=device)

        # ------------------------------------------------
        # CASE 3: Build training set composed of faces
        # ------------------------------------------------
        else:
            print("Correct single image detection")
            self.image_data(faces_dic, device=device)

        self.print_data_report(faces_dic=faces_dic, triplet=triplet_version)

        if save is not None:
            with open(save, 'wb') as output:
                try:
                    torch.save(self, output)  # , protocol=2) #pickle.HIGHEST_PROTOCOL
                    print("The set has been saved as " + save + "!\n")
                except MemoryError:
                    print("The dataset couldn't be saved")

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

    def build_triplet(self, faces_dic, device):
        """ ---------------------------------------------------
        Define the training set composed of (ref, pos, neg)
        List of 3 lists, each containing tensors

        !! Convention !!:
            Label = 0 if same people
            Label = 1 if different people
        --------------------------------------------------- """

        all_labels = list(faces_dic.keys())
        nb_labels = len(all_labels)

        # ================= Consider each person =================
        for label, pictures_list in faces_dic.items():
            # all_labels.remove(label)
            print("faces_dic" + str(faces_dic))
            labels_indexes_neg = [x for x in range(0, nb_labels) if x != all_labels.index(label)]
            print("labels_indexes_neg " + str(labels_indexes_neg))
            pos_pict_lists = []  # pictures_list

            # ================= Consider each picture of the person =================
            for i, picture_ref in enumerate(pictures_list):
                pos_pict_list = []
                pic_ind_pos = [j for j in range(len(pictures_list)) if j != i]
                nb_same_db = 0

                # ================= Consider several times the ref picture =================
                for j in range(NB_TRIPLET_PER_PICT):  # !! TO CHECK with very small db

                    # -------------- Positive Picture definition --------------
                    try:
                        curr_index_pos = random.choice(pic_ind_pos)

                        while curr_index_pos < i and i in pos_pict_lists[curr_index_pos]:
                            pic_ind_pos.remove(curr_index_pos)
                            curr_index_pos = random.choice(pic_ind_pos)

                    except IndexError:
                        # !! Means that the current label has no remaining other picture
                        # (empty sequence of available index)
                        break
                    picture_positive = pictures_list[curr_index_pos]
                    pic_ind_pos.remove(curr_index_pos)
                    pos_pict_list.append(curr_index_pos)  # Remember the pairs that are defined to avoid redundancy

                    # -------------- Negative Picture definition --------------
                    # Pick a random different person
                    curr_index_neg = random.choice(labels_indexes_neg)
                    label_neg = all_labels[curr_index_neg]
                    picture_negative = random.choice(faces_dic[label_neg])

                    if nb_same_db < NB_TRIPLET_PER_PICT / 2:  # Half of the negative must belong to the same db
                        nb_same_db += 1
                        try:
                            while 1 < len(self.all_db) and picture_negative.db_path != picture_ref.db_path:
                                labels_indexes_neg.remove(curr_index_neg)
                                # Pick a random different person
                                curr_index_neg = random.choice(labels_indexes_neg)
                                label_neg = all_labels[curr_index_neg]
                                picture_negative = random.choice(faces_dic[label_neg])
                        except IndexError:
                            break

                    # To keep track of the image itself in order to potentially print it
                    try:
                        self.train_not_formatted_data.append([picture_ref, picture_positive, picture_negative])
                        self.train_data.append([picture_ref.trans_img, picture_positive.trans_img,
                                                picture_negative.trans_img])

                        self.train_labels.append([0, 1])
                    except RuntimeError:  # RuntimeError:
                        print("ERR: In Build Triplet: Running out of Memory => Automatic Stop")
                        print("The current ref person is " + str(label))
                        print("Currently, " + str(len(self.train_data)) + " triplets have been defined")
                        break

                pos_pict_lists.append(pos_pict_list)

        # print("pos_pict_lists: " + str(pos_pict_lists))
        # self.train_data = torch.stack(self.train_data)
        self.train_labels = torch.tensor(self.train_labels)  # .to(device)

    """ --------------------- image_data ------------------------------------
      This function sets data by filling it with the pictures contained in 
      face_dic (that are the elements in the values)
      --------------------------------------------------------------------- """

    def image_data(self, faces_dic, device="cpu"):

        label_nb = 0
        # ========= Consider each person =================
        for label, pictures_list in faces_dic.items():
            # ======== Consider each picture of each person ==========
            for i, picture in enumerate(pictures_list):
                self.train_data.append(picture.trans_img)  # .to(device))
                self.train_labels.append(label_nb)
            label_nb += 1
        self.nb_classes = len(faces_dic)
        self.train_labels = torch.tensor(self.train_labels)  # .to(device)

    """ --------------------- to_single ------------------------------------
      This function returns a Face_DS object whose data are build from the 
      current data (that are triplets). 
      Basically, it takes each picture_ref and put it in the train_data of
      the Face_DS object to return. 
      --------------------------------------------------------------------- """

    def to_single(self, face_set):
        return Face_DS(face_set=face_set)

    """ ---------------------------------- print_data_report ------------------------------------  """

    def print_data_report(self, faces_dic=None, triplet=True):

        if faces_dic is None:
            print("\nThe total quantity of data is: " + str(2 * len(self.train_labels)))
            return

        # Report about the quantity of data
        pictures_nbs = [len(pictures_list) for person_name, pictures_list in faces_dic.items()]
        try:
            max_nb_pictures = max(pictures_nbs)
            min_nb_pictures = min(pictures_nbs)
        except ValueError:
            print("ERR: No data")
            raise ValueError

        print("\n ---------------- Report about the quantity of data  -------------------- ")
        if triplet:
            print("The total quantity of triplets used as data is: " + str(2 * len(self.train_labels)))
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
IN: db_list: list of db that were used 
OUT: list of 3 sets 
-------------------------------------------------------------- '''


def load_sets(db_name, dev, nb_classes, sets_list):
    type_ds = "triplet_" if nb_classes == 0 else "class" + str(nb_classes) + "_"
    result_sets_list = []
    save_names_list = ["trainset_ali_", "validationset_ali", "testset_ali"]
    for i, set in enumerate(sets_list):
        name_file = FOLDER_DB + save_names_list[i] + type_ds + db_name + ".pkl"
        try:
            with open(name_file, "rb") as f:
                loaded_set = torch.load(f)
                loaded_set.print_data_report(triplet=(nb_classes == 0))
                result_sets_list.append(loaded_set)
                print('Set Loading Success!\n')
        except (ValueError, IOError) as e:  # EOFError  IOError FileNotFoundError
            print("\nThe set " + name_file + " couldn't be loaded...")
            print("Building Process ...")
            #print("sset" + str(set.data_list))
            result_sets_list.append(Face_DS(fileset=set, device=dev, triplet_version=(nb_classes == 0), save=name_file))

        # ------- Classification Case: no testset -------
        if nb_classes != 0 and i == 1:
            break
    return result_sets_list


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

            new_data = Data(fn, ZIP_TO_PROCESS, True)
            dataset.add_data(new_data)

    dataset.ds_to_zip()


'''----------------------------- from_zip_to_data --------------------------------------------
 This function turns the content of the MAIN ZIP into a Fileset object
 !! 2 possible structures inside the main zip: 
 1. set of files whose name is like dbOrigin_person_id.jpg
 2. a folder containing folders (related to each person) containing files whose name is id.jpg
 ---------------------------------------------------------------------------------------------'''


def from_zip_to_data(with_profile, fname=MAIN_ZIP, dataset=None):
    t = time.time()
    if dataset is None: dataset = Fileset()
    if fname is None:
        fname = MAIN_ZIP

    print("\nData Loading from " + fname + " ...")
    with zipfile.ZipFile(fname, 'r') as archive:
        file_names = list(set(archive.namelist()))

        nb_entry = 0
        nb_entry_pers = 0
        previous_person = ""
        go_to_next_folder = False

        # ------ Create data from each image and add it to the dataset ----
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
            if fn[-1] == "/":
                go_to_next_folder = False
                continue
            if go_to_next_folder or fn == ".DS_Store":
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

            new_data = Data(fn, fname, False)

            if (with_profile or not new_data.lateral_face):
                dataset.add_data(new_data)

            # ------ Limitation of the total number of instances ------
            nb_entry += 1
            if MAX_NB_ENTRY < nb_entry:
                break

    print("Loading Time: " + str(time.time() - t))
    print(str(len(dataset.data_list)) + " pictures have been loaded!\n")
    return dataset


"""
This function generates synthetic images of each person contained in the given database
and add them in the database under the name "db__personName__indexReal_indexSynth.jpg"
IN: zip_db: Zip file containing the pictures the db is made up of  
"""


def generate_synthetic_im(db, nb_additional_images=Q_DATA_AUGM):
    # -------------------------------
    # 1. Open the DB
    # -------------------------------
    with zipfile.ZipFile(db, 'r') as archive:
        file_names = archive.namelist()
        try:
            file_names.remove(".DS_Store")
        except ValueError:
            pass

    face_dic = {}

    # ----------------------------------------------------------------------
    # 2. Go over each file and put 1 file per person as value
    # in a dictionary where the key is the person's name
    # ----------------------------------------------------------------------
    for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
        if db == FOLDER_DB + "FFHQ.zip":
            current_person = fn.split(".png")[0]
        else:
            current_person = fn.split(SEPARATOR)[1]
            if current_person in face_dic:
                continue

        pict = FaceImage(fn, None, db_path=db, i=fn.split(SEPARATOR)[2].split(".")[0])
        print("fn is " + str(fn))
        face_dic[current_person] = [pict]

    # ----------------------------------------------------------------------
    # 3. Generate synthetic images and add them in the dictionary
    # ----------------------------------------------------------------------
    print("In data Augmentation...")
    data_augmentation(face_dic, nb_add_instances=nb_additional_images, save=True)

    # -----------------------------------------------------------------
    # 4. Register the synthetic images in the db
    # -----------------------------------------------------------------
    fset = Fileset()

    for person, pictures_list in face_dic.items():
        for i, picture in enumerate(pictures_list):
            # fname = db__person__indexReal_indexSynth.jpg
            db_name = db.split("/")[-1].split("zip")[0]
            print("db name is " + db_name)
            print("person is " + person)
            index = str(picture.i) + "_" + str(1 + i)
            fname = db_name + SEPARATOR + person + SEPARATOR + index + "jpg"
            data = Data(fname, db, to_process=True, picture=picture)
            fset.add_data(data)

    #fset.ds_to_zip(db_destination=db + "_synthIm", file_is_pil=True) New db: Case of synthetic People
    fset.ds_to_zip(db_destination=db, file_is_pil=True)


"""
This function crops and aligns the images in db and registers them in 
"""


def register_aligned(db):
    # -------------------------------
    # 1. Open the DB
    # -------------------------------
    al_fset = Fileset()
    to_remove = []

    with zipfile.ZipFile(db, 'r') as archive:
        file_names = archive.namelist()
        try:
            file_names.remove(".DS_Store")
        except ValueError:
            pass

        # ----------------------------------------------------------------------
        # 2. Go over each file and put 1 file per person as value
        # in a dictionary where the key is the person's name
        # ----------------------------------------------------------------------
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
            original_image = Image.open(BytesIO(archive.read(fn))).convert("RGB")
            res_image = align_faces(original_image)
            if res_image is None:
                print("The file " + fn + " resulted in nontype transformation")
                to_remove.append(fn)
                continue
            data = Data(fn, db, to_process=True, picture=res_image)
            al_fset.add_data(data)

        # -----------------------------------------------------------------
        # 3. Register the aligned images in the db
        # -----------------------------------------------------------------
        al_fset.ds_to_zip(db_destination=db, file_is_pil=True)

        return to_remove


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    test_id = 2

    if test_id == 1:
        db_list = ["gbrieven", "cfp", "faceScrub", "lfw"]

        for i, db in enumerate(db_list):
            fileset = from_zip_to_data(False, fname=FOLDER_DB + db + ".zip")
            fileset.order_per_personName(TRANS, save=db)

    # TOTEST
    if test_id == 2:
        db_list = ["cfp70.zip"]  # , "FFHQ.zip"] #, "lfw.zip", "gbrieven.zip", "faceScrub.zip", "testdb.zip"]

        for i, db in enumerate(db_list):
            print("Current db is " + str(FOLDER_DB + db) + "...\n")
            generate_synthetic_im(FOLDER_DB + db)

    if test_id == 3:
        with zipfile.ZipFile(db, 'r') as archive:
            file_names = archive.namelist()
            try:
                file_names.remove(".DS_Store")
            except ValueError:
                pass

        face_dic = {}

        # ----------------------------------------------------------------------
        # 2. Go over each file and put 1 file per person as value
        # in a dictionary where the key is the person's name
        # ----------------------------------------------------------------------
        for i, fn in enumerate(file_names):  # fn = name[!]_id.extension
            if db == FOLDER_DB + "FFHQ.zip":
                current_person = fn.split(".png")[0]
            else:
                current_person = fn.split(SEPARATOR)[1]
                if current_person in face_dic:
                    continue

            face_dic[current_person] = ["coucou"]
            print("fn is " + fn)
