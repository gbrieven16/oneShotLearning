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

def rename_file(nameFolder):

    files = [str(os.path.join(dp, f)) for dp, dn, filenames in os.walk(nameFolder) for f in filenames]

    for i, file in enumerate(files):
        #new_name = "faces94_" + file.split(".")[0] + "_" + file.split(".")[1] + ".jpeg"
        try:
            new_name = nameFolder + file.split("faces94_")[1]
            os.rename(file, new_name)
        except IndexError:
            print("already")



if __name__ == "__main__":
    nameFolder = 'datasets/faces94/'
    rename_file(nameFolder)

