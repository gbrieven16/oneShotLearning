import zipfile
import os
from  PIL import Image
import sys
import shutil
from pathlib import Path


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


'''--------------------- data_to_zip ------------------------------------------ '''


def rename_file(nameFolder, zip_name=None):
    files = [str(os.path.join(dp, f)) for dp, dn, filenames in os.walk(nameFolder) for f in filenames]

    for i, file in enumerate(files):
        print("file is " + file)
        # new_name = "faces94_" + file.split(".")[0] + "_" + file.split(".")[1] + ".jpeg"
        try:
            new_name = file.replace("_", "__")
            os.rename(file, new_name)
        except IndexError:
            print("already")
    if zip_name is not None:
        shutil.make_archive(zip_name, 'zip', nameFolder)


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
        #new_name = "facescrub/" + fn.split("/")[0] + "/" + fn.split("/")[2]
        #zipf.write(root_dir + "facescrub/" + fn, new_name)
        zipf.write(fn)

    zipf.close()



if __name__ == "__main__":
    #data_to_zip()
    nameFolder = 'data/gbrieven/gbrieven/'
    rename_file(nameFolder, zip_name="gbrieven")


