import zipfile
import os
from  PIL import Image
import sys

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
            new_im.save('foo'+str(i)+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence


if __name__ == "__main__":
    Image.open('subject01.glasses').convert('RGB').save('prev.jpeg')
    #processImage('subject01.glasses')