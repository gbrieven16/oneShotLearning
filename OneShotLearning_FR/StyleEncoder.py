import time
import sys
import os
import pickle
import platform
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
To generate new data, 2 approaches can be used: 
1. Give an image and generate several instances of this image 
2. Generate synthetic faces and then produce several instances from 
"""

#########################################
#       GLOBAL VARIABLES                #
# sys.argv[1]: IMAGE_SIZE               #
# sys.argv[2]: INDEX_VEC_CHANGE         #
#########################################

GENERATED_IMAGES_DIR = "/home/gbrieven/datasets/synth_im/"
FROM_ROOT = "" if platform.system() == "Darwin" else "/home/gbrieven/oneShotLearning/OneShotLearning_FR/"
DIRECTION_DIR = FROM_ROOT + 'ffhq_dataset/latent_directions/'
DLATENT_DIR = "/home/gbrieven/datasets/latent_repres/" if platform.system() != "Darwin" \
    else "data/gbrieven/latent_repres/"


URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
STYLE_GAN = FROM_ROOT + "models/karras2019stylegan-ffhq-1024x1024.pkl"

BATCH_SIZE = 1  # If more than 1, then mix of faces

try:
    IMAGE_SIZE = int(sys.argv[1])
except IndexError:
    IMAGE_SIZE = 256

try:
    INDEX_VEC_CHANGE = int(sys.argv[2])
except IndexError:
    INDEX_VEC_CHANGE = None

LR = 1  # CHANGE Was set to 1
NB_ITERATIONS = 1200  # !!! the higher it is, the more similar to the given input data the generated image is
RANDOMIZE_NOISE = False

CHANGES = ["smile", "age", "gender"]
COEF = {"smile": [-1.5, 0, 1.3], "age": [-1.5, 0], "gender": [-1, 0]}


if False and platform.system() != "Darwin":
    if GENERATED_IMAGES_DIR is not None:
        try:
            # ----- Define Directories if they don't exist ----------
            os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
            os.makedirs(DLATENT_DIR, exist_ok=True)
        except (PermissionError, OSError) as e:
            print("IN STYLE ENCODER: Directories couldn't be created: do it manually! \n")

    print('Memory assigned to STYLE GAN! \n')
    tflib.init_tf()  # Initialization of TensorFlow session
    _, _, GS_NETWORK = pickle.load(open(STYLE_GAN, "rb"))  # generator_network, discriminator_network
    GENERATOR = Generator(GS_NETWORK, BATCH_SIZE, randomize_noise=RANDOMIZE_NOISE)

    PERCEPTUAL_MODEL = PerceptualModel(IMAGE_SIZE, layer=9, batch_size=BATCH_SIZE)
    PERCEPTUAL_MODEL.build_perceptual_model(GENERATOR.generated_image)
else:
    GS_NETWORK = None


#########################################
#       Functions                       #
#########################################
def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


""" ---------- print_network_architecture ---------------------------
IN: Gs: dnnlib.tflib.network.Network object 
------------------------------------------------------------------ """


def print_network_architecture(Gs):
    # Print network details.
    print(" =================== Structure of the Generator StyleGAN ==================== ")
    Gs.print_layers()


""" -------------------- generate_face --------------------------------
This function generates synthetic face images 
IN: Gs: Network 
---------------------------------------------------------------------- """


def generate_synth_face(Gs=GS_NETWORK, save=True, nb_people=1):
    # Pick latent vector => Artificial vector respecting some distribution constraints

    for i in range(nb_people):
        rnd = np.random.RandomState(i)
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate face image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        if save:
            filename = GENERATED_IMAGES_DIR + 'synthDB__synthPers' + "_" + str(i) + "__1.jpg"
            PIL.Image.fromarray(images[0], 'RGB').save(filename)
            print("Synthetic image saved as " + filename + "\n")


def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    GENERATOR.set_dlatents(latent_vector)
    img_array = GENERATOR.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((IMAGE_SIZE, IMAGE_SIZE))


""" ------------------- move_and_show -----------------------------------------
This function produces a synthetic image resulting from a latent change 
applied to the input latent representation 
IN: latent_representation: numpy.ndarray resulting from a face picture 
-------------------------------------------------------------------------------- """


def move_and_show(latent_vector, direction, coeff, save_result):
    if INDEX_VEC_CHANGE is None:
        new_latent_vector = (latent_vector + coeff * direction)
    else:
        new_latent_vector = latent_vector.copy()
        new_latent_vector[INDEX_VEC_CHANGE] = (latent_vector + coeff * direction)[INDEX_VEC_CHANGE]
    synthetic_image = generate_image(new_latent_vector)
    #try:
    plt.imshow(synthetic_image)
    plt.title('Coeff: %0.1f' % coeff)
    plt.show()
    if save_result is not None:
        plt.savefig(save_result)
        print("The set of synthetic images has been saved!")
    #except:
        #pass
    return synthetic_image


""" ------------------- apply_latent_direction --------------------------------
This function produces a synthetic image resulting from a latent change 
applied to the input latent representation 
IN: latent_representation: numpy.ndarray resulting from a face picture 
-------------------------------------------------------------------------------- """


def apply_latent_direction(latent_representation, direction="smile", coef=0, save_result=None):
    # Direction Loading ...
    direction = np.load(DIRECTION_DIR + direction + '.npy')

    # Face Change Application
    return move_and_show(latent_representation, direction, coef, save_result)


""" ------------------------------------ get_encoding ----------------------------------------------------
This function retrains the model from the new data of the given training set 
IN: src_dir: directory containing the training data 
    generated_images_dir: dir where to store synthetic image generated by the generator during training
    dlatent_dir: dir where to store the FR of the training pictures  
OUT: list of the dlatent representation of the pictures contained in src dir 
     list of lists of names (where one list corresponds to one batch) 
--------------------------------------------------------------------------------------------------------- """


def get_encoding(db_source, filename, generated_images_dir=None, dlatent_name=None, with_save=False):
    try:
        return np.load(DLATENT_DIR + dlatent_name)
    except (FileNotFoundError, TypeError) as e:
        time_init = time.time()
        print("\nOptimize (only) dlatents by minimizing perceptual loss between reference and generated images in "
              "feature space... ")

        for images_batch in tqdm(split_to_batches([filename], BATCH_SIZE), total=1 // BATCH_SIZE):

            PERCEPTUAL_MODEL.set_reference_images(images_batch, zip_file=db_source)
            op = PERCEPTUAL_MODEL.optimize(GENERATOR.dlatent_variable, iterations=NB_ITERATIONS, learning_rate=LR)
            pbar = tqdm(op, leave=False, total=NB_ITERATIONS)
            for loss in pbar:
                pbar.set_description(' '.join([filename]) + ' Loss: %.2f' % loss)
            print(' '.join([filename]), ' loss:', loss)

            # ---- Generate images from found dlatents (and save them) ... -----
            generated_images = GENERATOR.generate_images()
            generated_dlatents = GENERATOR.get_dlatents()  # of type'numpy.ndarray' 1x18x512 (= 9216 elements)

            # ----------------------
            #       Saving
            # ----------------------

            if with_save:
                if generated_images_dir is not None:
                    img = PIL.Image.fromarray(generated_images, 'RGB')
                    img.save(os.path.join(generated_images_dir, filename), 'jpeg')
                if dlatent_name is not None:
                    np.save(DLATENT_DIR + dlatent_name, generated_dlatents)
                    print(dlatent_name + " has been saved!")

            GENERATOR.reset_dlatents()

        print("Time for encoding is " + str(time.time() - time_init) + "\n")
        return generated_dlatents


""" ------------------------------------ data_augmentation ----------------------------------------------------
This function adds to the dictionary synthetic images representing the key person
IN: face_dic: dictionary where the key is the name of a person and the value is a list of FaceImage corresponding 
              to the person 
    images: list of face pictures 
    nb_add_instances: number of additional instances to produce per person 
    
"REMINDER":FaceImage:         
        self.file_path = file_path      # Complete path of the image
        self.db_path = db_path   # Complete path of the db (zip file)
        self.trans_img = trans_image
        self.dist = {} # key1 person, val1: dic2 ; key2: index, val2: val2
        self.feature_repres = None
        self.person = pers
        self.index = i

REM: IMPL_CHOICE: the latent representation related to a person is derived from their first picture (only) 
----------------------------------------------------------------------------------------------------------------- """


def data_augmentation(face_dic=None, nb_add_instances=3, save_generated_im=False, save_dlatent=False, dirs=None):

    if nb_add_instances == 0:
        return

    if dirs is None:
        dirs = CHANGES

    for person, images in face_dic.items():
        if save_generated_im:
            images[0].save_im(GENERATED_IMAGES_DIR + person + ".jpg")

        if save_dlatent:
            dlatent_name = images[0].file_path.split(".")[0] + ".npy"
        else:
            dlatent_name = None

        # Get latent representation of the person
        latent_repres = get_encoding(images[0].db_path, images[0].file_path, dlatent_name=dlatent_name, with_save=True)
        nb_additional_pict = 0

        time_in = time.time()
        for i, change in enumerate(dirs):
            for j, coef in enumerate(COEF[change]):

                if nb_additional_pict < nb_add_instances:
                    nb_additional_pict += 1
                else:
                    break

                new_image = apply_latent_direction(latent_repres, direction=change, coef=coef)  # save_result=...

                # Add the new image in the dictionary
                face_dic[person].append(new_image)

                if save_generated_im:
                    name = GENERATED_IMAGES_DIR + change + "_" + str(coef) + "_" + person + "__" + str(i) + str(j)
                    new_image.save(name + ".jpg", "jpeg")
                    print("Synthetic image saved as " + name + ".jpg" + "\n")

        print("Time to generate " + str(nb_add_instances) + " pictures is " + str(time.time() - time_in) + "\n")


if __name__ == "__main__":
    # Test image generation
    images = ["testdb__0000107__2.jpg", "testdb__0000159__16.jpeg"]

    for j, image in enumerate(images):
        z = get_encoding("/home/gbrieven/datasets/" + "testdb_filtered.zip", image)
        print(z)
        change = "smile"
        for i, coef in enumerate(COEF["smile"]):
            save_result = GENERATED_IMAGES_DIR + change + "_" + str(coef) + "_" + image.split(".")[0]
            save_result += str(IMAGE_SIZE) + "_nbElem" + str(INDEX_VEC_CHANGE) + ".jpeg"
            apply_latent_direction(z, direction=change, coef=coef)

