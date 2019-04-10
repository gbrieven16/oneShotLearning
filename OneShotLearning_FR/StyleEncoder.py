import os
import pickle
import platform
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
import config
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
#########################################

GENERATED_IMAGES_DIR = "/data/gbrieven/synth_im/"
DLATENT_DIR = "/data/gbrieven/latent_repres/"

if GENERATED_IMAGES_DIR is not None:
    try:
        # ----- Define Directories if they don't exist ----------
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
        os.makedirs(DLATENT_DIR, exist_ok=True)
    except PermissionError:
        print("IN STYLE ENCODER:Directories couldn't be created: do it manually! \n")

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
STYLE_GAN = "models/karras2019stylegan-ffhq-1024x1024.pkl"
BATCH_SIZE = 1  # If more than 1, then mix of faces
NB_PICTURES = 3
IMAGE_SIZE = 256
LR = 1  # CHANGE Was set to 1
NB_ITERATIONS = 1400  # !!! the higher it is, the more similar to the given input data the generated image is
RANDOMIZE_NOISE = False

CHANGES = ["smile", "age", "gender"]
COEF = {"smile": [-1, 0, 1], "age": [-1, 0], "gender": [-1, 0]}
# COEF = [-1, 0, 1]  # Coefficient measuring the intensity of change

if platform.system() != "Darwin":
    tflib.init_tf()  # Initialization of TensorFlow session
    _, _, GS_NETWORK = pickle.load(open(STYLE_GAN, "rb"))  # generator_network, discriminator_network
    GENERATOR = Generator(GS_NETWORK, BATCH_SIZE, randomize_noise=RANDOMIZE_NOISE)

    PERCEPTUAL_MODEL = PerceptualModel(IMAGE_SIZE, layer=9, batch_size=BATCH_SIZE)
    PERCEPTUAL_MODEL.build_perceptual_model(GENERATOR.generated_image)


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


def generate_synth_face(Gs, save=True, nb_people=1):
    # Pick latent vector => Artificial vector respecting some distribution constraints

    for i in range(nb_images):
        rnd = np.random.RandomState(i)
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate face image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        if save:
            os.makedirs(config.result_dir, exist_ok=True)
            filename = os.path.join(config.result_dir, 'synthImage_m2_' + str(i) + '.jpeg')
            print("Saving of image in location " + filename)
            PIL.Image.fromarray(images[0], 'RGB').save(filename)


def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    GENERATOR.set_dlatents(latent_vector)
    img_array = GENERATOR.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))


""" ------------------- move_and_show -----------------------------------------
This function produces a synthetic image resulting from a latent change 
applied to the input latent representation 
IN: latent_representation: numpy.ndarray resulting from a face picture 
-------------------------------------------------------------------------------- """


def move_and_show(latent_vector, direction, coeff, save_result):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
    synthetic_image = generate_image(new_latent_vector)
    plt.imshow(synthetic_image)
    plt.title('Coeff: %0.1f' % coeff)
    plt.show()
    if save_result is not None:
        plt.savefig(save_result)
    return synthetic_image


""" ------------------- apply_latent_direction --------------------------------
This function produces a synthetic image resulting from a latent change 
applied to the input latent representation 
IN: latent_representation: numpy.ndarray resulting from a face picture 
-------------------------------------------------------------------------------- """


def apply_latent_direction(latent_representation, direction="smile", coef=0, save_result=None):
    # Direction Loading ...
    direction = np.load('ffhq_dataset/latent_directions/' + direction + '.npy')

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
    print("Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in "
          "feature space... \n")

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
            img = PIL.Image.fromarray(generated_images, 'RGB')
            if generated_images_dir is not None:
                img.save(os.path.join(generated_images_dir, f'{filename}.jpeg'), 'jpeg')
            if dlatent_name is not None:
                np.save(dlatent_name, generated_dlatents)
                print(dlatent_name + " has been saved!")

        GENERATOR.reset_dlatents()

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


def data_augmentation(face_dic=None, nb_add_instances=3, save=False):
    if nb_add_instances == 0:
        return

    for person, images in face_dic.items():
        if save:
            images[0].save_im(GENERATED_IMAGES_DIR + person + ".jpg")

        # Get latent representation of the person
        latent_repres = get_encoding(images[0].db_path, images[0].file_path)
        nb_additional_pict = 0

        for i, change in enumerate(CHANGES):
            for j, coef in enumerate(COEF[change]):

                if nb_additional_pict < nb_add_instances:
                    nb_additional_pict += 1
                else:
                    break

                new_image = apply_latent_direction(latent_repres, direction=change, coef=coef)  # save_result=...
                print("New image is " + str(new_image) + ' and the type is ' + str(type(new_image)))

                # Add the new image in the dictionary
                face_dic[person].append(new_image)

                if save:
                    name = GENERATED_IMAGES_DIR + person + "__" + str(i) + str(j) + ".jpg"
                    new_image.save(name, "jpeg")
                    print("Synthetic image saved as " + name + "\n")


if __name__ == "__main__":
    pass
