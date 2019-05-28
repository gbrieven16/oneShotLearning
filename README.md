# Face Recognition Using One-shot Learning

 [![Generic badge](https://img.shields.io/badge/anaconda-4.6.14-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/pytorch-1.1.0-<COLOR>.svg)](https://shields.io/)  [![Generic badge](https://img.shields.io/badge/tensorflow-1.13.1-<COLOR>.svg)](https://shields.io/) 

This repository contains a pytorch implementation of a face recognition system relying on a Siamese Network. This solution contains 3 main steps:

  - Data Processing
  - Siamese Network Training 
  - Classification (where the Face Recognition task itself is performed)

In addition to those main steps, first, a data augmentation process has been defined relying on the generation of synthetic data. To perform this, a Style GAN is used, implemented in tensorflow and already trained. Next, a pretraining phase has been defined, where an autoencoder is trained on face images. By doing so, the encoder part can be retrained as part of the Siamese Network.

All around this solution, some code is also dedicated to experiment, visualize and interprete multiple scenarios. Those multiple scenarios are mainly defined through the global variables set on top of the different scripts (referring to the number of epochs for training, the quantity of data to use, the distance metric to employ...).

A very broad description of what's implmented here is given below. 

### Data Augmentation 

Before the data processing phase, additional synthetic data can be generated from the real ones by using the style GAN implementation provided through https://github.com/Puzer/stylegan-encoder. In this way:

- Synthetic people can be defined 
- Synthetic additional instances related to a real person can be defined 

All this data augmentation part is supported by the *StyleEncoder.py* script, relying on the encoder, the dnnlib and the ffhq_dataset packages coming directly from the style GAN implementation.

### Data Processing
During this phase, first, the image data are processed, being aligned, cropped and turned into a pytorch tensor. Then the resulting tensor is normalized. 

Once processed, the data are ordered by person (since a data results from a face picture) and used to build triplets (A,P,N) composing the training, validation and testing datasets. A given triplet is such that: 
- A is the Anchor (i.e. the "reference picture") 
- P is the Positive (i.e. a picture representing the same person as A) 
- N is the Negative (i.e. a picture representing a person different from A)

Some constraints can be imposed on those triplets, like exclusively using real images, selecting a Negative coming from the same database as the Anchor, builing n triplets related to each Anchor, imposing the same number of pictures per person .... 

All this processing part is supported by the *Dataprocessing.py* and the *FaceAlignment.py* scripts. 

### Siamese Network Training 

Once processed, the embedding network belonging to the Siamese Netork may be trained as the encoder of an autoencoder, taking as input the anchor of each triplet, to get initialized its weights. Next, the Siamese Network is trained, directed by the triplet loss function. 
Notice that other loss functions are also implemented and can be experimented, like the contrastive loss or the cross entropy loss. 

All the training part is supported by the script *Model.py*. Besides this, the global structure of the Siamese Network is implemented in *NeuralNetwork.py*, where the Autoencoder class and different classes related to each loss are defined. Finally, regarding the architecture of the embedding network, all of them are implemented in *EmbeddingNetwork.py* and implemented with the help of some external code:

|         | Architecture           | Source  |
| ------------- |:-------------:| -----:|
| 1. | BasicNet | https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18 |
| 2. | AlexNet  | https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py |
| 3. | VGG16    | https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py     |

To get the best possible model, an experimentation phase has been designed in *Experimentation.py*, relying on the definition of different scenarios, differing from each others in terms of nature and quantity of data, learning rate, losses ...

### Classification

To perform the face recognition task, 2 main components are defined:
- a gallery composing of identitied face  
- a set of probes where the target is getting the identity of each them

To do so, for a given probe, its feature representation is derived, after being propagated through the Siamese Network defined below, and compared to the ones of each instance of each person in the gallery. After that, the identities in the gallery can be ranked according to the degree of similarity evaluated over the comparison process and the top identity is predicted and assigned to the probe. This is implemented in *FaceRecognition.py*. 
