import zipfile
import os
from  PIL import Image
import sys
import shutil


def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save('foo' + str(i) + '.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass  # end of sequence


'''--------------------- extract_files --------------------------------
 This function extracts from the main folder all the files 
 IN: Name of the main folder
 --------------------------------------------------------------------------'''


def extract_files(nameFolder):
    source = [str(os.path.join(dp, f)) for dp, dn, filenames in os.walk(nameFolder) for f in filenames]
    destination = nameFolder

    for files in source:
        if files.endswith(".jpg"):
            try:
                shutil.move(files, destination)
            except shutil.Error:
                print("Already here: " + files.split(nameFolder)[1])


'''--------------------- rename_file ------------------------------------------ '''


def rename_file(nameFolder, zip_name=None):
    files = [str(os.path.join(dp, f)) for dp, dn, filenames in os.walk(nameFolder) for f in filenames]
    print("The files are " + str(files))
    people = {}

    for i, file in enumerate(files):

        print("file is " + file)
        person = file.split("/")[3]

        if person not in people:
            people[person] = 0
        else:
            people[person] += 1

        new_name = nameFolder + zip_name + "__" + person + "__" + str(people[person]) + ".jpg"
        print("The new name is: " + str(new_name))

        try:
            os.rename(file, new_name)
        except IndexError:
            print("already")

    if zip_name is not None:
        shutil.make_archive(zip_name, 'zip', nameFolder)
        print("Archive has been made!")


'''--------------------- data_to_zip ------------------------------------------
 This function zip the content of a directory, excluding the directory 
 "profile". 
 root_dir is contains: namePerson/profile/id.jpg and namePerson/frontal/id.jpg
 !!! The initial zip has to be decompressed !!!
 ------------------------------------------------------------------------------'''


def data_to_zip(root_dir):
    # ---- Get all the relative path from given directory ------
    file_names = []

    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)
            file_names.append(rel_file)

    zipf = zipfile.ZipFile("data/gbrieven/gbrieven.zip", 'w', zipfile.ZIP_DEFLATED)

    for i, fn in enumerate(file_names):
        print("fn is " + str(fn))
        if fn == "./.DS_Store" or fn == "__MACOSX/" or fn[-1] == "/":
            continue

        if fn.split("/")[1] == "profile":
            continue
        # new_name = "facescrub/" + fn.split("/")[0] + "/" + fn.split("/")[2]
        # zipf.write(root_dir + "facescrub/" + fn, new_name)
        zipf.write(fn)

    zipf.close()


'''--------------------- delete_from_zip ------------------------------------------
 This function "removes" the files whose name is in the to_remove_list by creating
 a new zip file  
 --------------------------------------------------------------------------------'''


def delete_from_zip(zip_file_in, to_remove_list, zip_file_out):
    zin = zipfile.ZipFile(zip_file_in, 'r')
    zout = zipfile.ZipFile(zip_file_out, 'w')

    for item in zin.infolist():
        buffer = zin.read(item.filename)
        if item.filename.split(".") != "jpeg": # not in to_remove_list:
            print("file ok  " + str(item.filename))
            zout.writestr(item, buffer)
    zout.close()
    zin.close()


# !! Ou faire un truc récursif
def pickle_big(dict, save):
    FOLDER_DB = "data/gbrieven"
    not_save = True
    divisor = 1
    while not_save:
        try:
            # trans = 1
            # faces_dic1 = {k: v for k, v in faces_dic.items() if k in list(faces_dic)[:20]}
            #
            transition = int(round(len(dict) / divisor))
            start = 0
            for i in range(divisor):
                dict = {k: v for k, v in dict.items() if k in list(dict)[start:(i + 1) * transition]}
                pickle.dump(dict, open(FOLDER_DB + "faceDic_" + save + str(i) + ".pkl", "wb"))
                start = (i + 1) * transition
            not_save = False
        except OSError:
            divisor *= 2


if __name__ == "__main__":
    test_id = 1

    if test_id == 1:
        shutil.make_archive("data/data_augm", 'zip', "data/gbrieven/data_augm")
        print("Archive has been made!")

    if test_id == 2:
        # data_to_zip()
        nameFolder = 'data/gbrieven/cfp/'
        zip_name = "cfp"
        # rename_file(nameFolder, zip_name=zip_name)
        #file_to_remove = "cfp__003__10.jpg"
        zip_file_in = "data/gbrieven/gbrieven.zip"
        zip_file_out = "data/gbrieven.zip"
        delete_from_zip(zip_file_in, None, zip_file_out)
