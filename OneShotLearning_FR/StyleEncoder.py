import os
import pickle
import zipfile
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
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

GENERATED_IMAGES_DIR = "/data/gbrieven/FFHQ500_generated"
DLATENT_DIR = "/data/gbrieven/FFHQ500_latent"

if generated_images_dir is not None:
    # ----- Define Directories if they don't exist ----------
    os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
    os.makedirs(DLATENT_DIR, exist_ok=True)

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
BATCH_SIZE = 1  # If more than 1, then mix of faces
NB_PICTURES = 3
IMAGE_SIZE = 256
LR = 1  # CHANGE Was set to 1
NB_ITERATIONS = 1000  # !!! the higher it is, the more similar to the given input data the generated image is
RANDOMIZE_NOISE = False

CHANGES = ["smile", "gender", "age"]
COEF = [-1, 0, 2]  # Coefficient measuring the intensity of change

tflib.init_tf()  # Initialization of TensorFlow session
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    _, _, GS_NETWORK = pickle.load(f)  # generator_network, discriminator_network

GENERATOR = Generator(GS_NETWORK, BATCH_SIZE, randomize_noise=RANDOMIZE_NOISE)


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


def get_encoding(db_source, filename, perceptual_model, generated_images_dir=None, dlatent_dir=None):
    print("Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in "
          "feature space... \n")

    perceptual_model.set_reference_images(filename, zip_file=db_source)
    op = perceptual_model.optimize(GENERATOR.dlatent_variable, iterations=NB_ITERATIONS, learning_rate=LR)
    pbar = tqdm(op, leave=False, total=NB_ITERATIONS)
    for loss in pbar:
        pbar.set_description(' '.join(names) + ' Loss: %.2f' % loss)
    print(' '.join(names), ' loss:', loss)

    # ---- Generate images from found dlatents (and save them) ... -----
    generated_images = GENERATOR.generate_images()
    generated_dlatents = GENERATOR.get_dlatents()  # of type'numpy.ndarray' 1x18x512 (= 9216 elements)

    if with_save:
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(generated_images_dir, f'{img_name}.jpeg'), 'jpeg')
            np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), dlatent)

    GENERATOR.reset_dlatents()

    return generated_dlatents


"""
This function adds to the dictionary synthetic images representing the key person 
IN: face_dic: dictionary where the key is the name of a person and the value is a list of FaceImage corresponding 
              to the person 
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
"""


def data_augmentation(face_dic, nb_add_instances=3):
    if nb_add_instances == 0:
        return

    # print_network_architecture(GS_NETWORK)
    perceptual_model = PerceptualModel(IMAGE_SIZE, layer=9, batch_size=BATCH_SIZE)
    perceptual_model.build_perceptual_model(GENERATOR.generated_image)

    for person, images in face_dic.items():
        # Get latent representation of the person
        latent_repres = get_encoding(images[0].db_path, images[0].file_path, perceptual_model)
        nb_additional_pict = 0

            for i, change in enumerate(CHANGES):
                for j, coef in enumerate(COEF):

                    if nb_additional_pict < nb_add_instances:
                        nb_additional_pict += 1
                    else:
                        break

                    new_image = apply_latent_direction(latent_repres, direction=change, coef=coef)
                    print("New image is " + str(new_image) + ' and the type is ' + str(type(new_image)))

                    # Add the new image in the dictionary

