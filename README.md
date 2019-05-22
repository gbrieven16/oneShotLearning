# Face Recognition Using One-shot Learning

[![Python 3.6](https://img.shields.io/badge/python-3.6.8-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Generic badge](https://img.shields.io/badge/anaconda-4.6.14-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/pytorch-1.1.0-<COLOR>.svg)](https://shields.io/)  [![Generic badge](https://img.shields.io/badge/tensorflow-1.13.1-<COLOR>.svg)](https://shields.io/) 

This repository contains a pytorch implementation of a face recognition system relying on a Siamese Network. This solution contains 3 main steps:

  - Data Processing
  - Siamese Network Training 
  - Face Recognition task

In addition to those main steps, first, a data augmentation process has been defined relying on the generation of synthetic data. To perform this, a Style GAN is used, implemented in tensorflow and already trained. Next, a pretraining phase was defined, where an autoencoder is trained on face images. By doing so, the encoder part can be retrained as part of the siamese network.

All around this solution, some code is also dedicated to experiment, visualize and interprete multiple scenarios. Those multiple scenarios are mainly defined through the global variables set in the different scripts (referring to the number of epoch for training, the quantity of data to use, the distance metric to employ...)

### Data Augmentation 

Before the data processing phase, additional synthetic data can be generated from the real ones by using the style GAN implementation provided through https://github.com/Puzer/stylegan-encoder. In this way:

- Synthetic people can be defined 
- Synthetic additional instances related to a real person can be defined 

All this data augmentation part is supported by the StyleEncoder.py script, relying on the encoder package coming directly from the style GAN implementation.

### Data Processing
During this phase, first, the image data are processed, being aligned, cropped and turned into a pytorch tensor. Then the resulting tensor is normalized. 

Once processed, the data are ordered by person (since a data results from a face picture) and used to build triplets (A,P,N) composing the training, validation and testing datasets. A given triplet is such that: 
- A is the Anchor (i.e. the "reference picture") 
- P is the Positive (i.e. a picture representing the same person as A) 
- N is the Negative (i.e. a picture representing a person different from A)

All this processing part is supported by the Dataprocessing.py and the FaceAlignment.py scripts. 

### Siamese Network Training 

Once processed, the embedding network belonging to the siamese netork may be trained as the encoder of an autoencoder, taking as input the anchor of each triplet. Next, the siamese network is trained, directed by the triplet loss function. Notice that other loss functions are also implemented and can be experimented. 

All the training part is supported by the script Model.py. Besides this, the global structure of the siamese network is implemented in NeuralNetwork.py, where the Autoencoder class and different classes related to each loss are defined. Finally, regarding the structure of the embedding network, all the different architectures were implemented in EmbeddingNetwork.py.

### Face Recognition Task 

To perform the face recognition task, a gallery and a set of probes are defined and the target is getting the identity of each probe by comparing it to each instance of each person in the gallery. This is implemented in FaceRecognition.py. 
