import torch
import os
import pickle
from NeuralNetwork import Classif_Net, AutoEncoder, TYPE_ARCH
from Visualization import visualization_test, visualization_train
from Dataprocessing import Face_DS
from torch import nn
from torch import optim

#########################################
#       GLOBAL VARIABLES                #
#########################################


MARGIN = 2.0
MOMENTUM = 0.9
N_TEST_IMG = 5
PT_BS = 32  # Batch size for pretraining
PT_NUM_EPOCHS = 200

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loss combination not considered here


# ================================================================
#                    CLASS: Classifier
#  ================================================================

class Classifier:
    def __init__(self, train_param=None, test_loader=None, embedding_net=None, network=None, train_loader=None,
                 nb_classes=None):

        # Default Initialization
        self.network = network
        self.loss_type = "None"
        self.train_loader = train_loader
        self.lr = 0.001
        self.wd = 0.001
        self.optimizer = None
        self.class_weights = [1, 1]

        if train_param is not None:
            self.set_for_training(train_param, embedding_net, nb_classes=nb_classes)

        self.test_loader = test_loader  # No need if autoencoder training

        self.eval_dic = {"nb_correct": 0, "nb_labels": 0}

        # For Visualization
        self.losses_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.f1_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.losses_train = []

    '''------------------------ set_for_training -------------------------------- '''
    def set_for_training(self, train_param, embedding_net, nb_classes):

        # ----------------- Network Definition -------------
        if train_param["loss_type"] is None:
            self.network = AutoEncoder(embedding_net)
        elif train_param["loss_type"] == "ce_classif":
            self.network = Classif_Net(nb_classes=nb_classes)

        self.loss_type = train_param["loss_type"]

        # ----------------- Optimizer Definition -------------
        try:
            self.lr = train_param["hyper_par"]["lr"]
            self.wd = train_param["hyper_par"]["wd"]
        except KeyError:
            pass  # We keep the default setting

        try:
            self.optimizer = self.get_optimizer(opt_type=train_param["opt_type"])
        except KeyError:
            pass  # We keep the default setting

        # ----------------- Class Weighting Use -------------
        try:
            self.weighted_classes = train_param["weighted_class"]
        except KeyError:
            pass  # We keep the default setting

    '''------------------------ pretraining ----------------------------------------------
       The function trains an autoencoder based on given training data 
       IN: train_data: Face_DS objet whose training data is a list of 
           face images represented through tensors 
           autocoder: an autocoder characterized by an encoder and a decoder 
    ------------------------------------------------------------------------------------ '''

    def pretraining(self, training_set, num_epochs=PT_NUM_EPOCHS, batch_size=PT_BS, loss_type=None):

        try:
            with open("autoencoder.pkl", "rb") as f:
                self.network.embedding_net = pickle.load(f)
        except (IOError, FileNotFoundError) as e:
            train_data = Face_DS(training_set, device=DEVICE, triplet_version=False)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            train_param = {"loss_type": loss_type, "opt_type": "Adam"}

            autoencoder = Classifier(train_param, embedding_net=self.network.embedding_net, train_loader=train_loader)

            print(" ------------ Train as Autoencoder ----------------- ")
            for epoch in range(num_epochs):
                autoencoder.train(epoch, autoencoder=True)
            autoencoder.network.visualize_dec()

            pickle.dump(self, open("autoencoder.pkl", "wb"), protocol=2)
            print("The set has been saved!\n")

    '''------------------ train_nonpretrained -------------------------------------
       The function trains a neural network (so that it's performance can be 
       compared to the ones of a NN that was pretrained) 
    -------------------------------------------------------------------------------- '''

    def train_nonpretrained(self, num_epochs, optimizer_type):
        train_param = {"train_loader": self.train_loader, "loss_type": self.loss_type, "opt_type": optimizer_type}
        model_comp = Classifier(train_param, test_loader=self.test_loader)

        for epoch in range(num_epochs):
            print("-------------- Model that was not pretrained ------------------")
            model_comp.train(epoch)
            loss_notPret, f1_notPret = model_comp.test()
            self.losses_test["Non-pretrained Model"].append(loss_notPret)
            self.f1_test["Non-pretrained Model"].append(f1_notPret)

    '''---------------------------- train --------------------------------
     This function trains the network attached to the model  
     -----------------------------------------------------------------------'''

    def train(self, epoch, autoencoder=False):

        self.network.train()
        loss_list = []
        # ------- Go through each batch of the train_loader -------
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()  # clear gradients for this training step

            for i in range(len(data)):  # List of 3 tensors
                data[i] = data[i].to(DEVICE)

            try:
                if autoencoder:
                    # ----------- CASE 1: Autoencoder Training -----------
                    data_copy = data  # torch.unsqueeze(data, 0)
                    encoded, decoded = self.network(data)
                    loss = nn.MSELoss()(decoded, data_copy)  # mean square error
                else:
                    # ----------- CASE 2: Image Differentiation Training -----------
                    loss = self.network.get_loss(data, target, self.class_weights)

            except IOError:  # The batch is "not complete"
                break

            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            loss_list.append(loss.item())

            # -----------------------
            #   Visualization
            # -------------------------
            if batch_idx % 10 == 0:
                if autoencoder:
                    batch_size = len(data)
                else:
                    batch_size = len(data[0])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(self.train_loader.dataset),
                           100. * batch_idx * batch_size / len(self.train_loader.dataset),
                    loss.item()))  # len(data[0]) = batch_size

        self.losses_train.append(loss_list)

    '''---------------------------- test --------------------------------
     This function tests the given model 
     OUT: loss: current loss once evaluated on the testing set
          f1-measure: current f1-measure once evaluated on the testing set
     -----------------------------------------------------------------------'''

    def test(self, for_weight_update=False):
        self.network.eval()
        self.reset_eval()

        with torch.no_grad():
            loss_test = 0

            data_loader = self.train_loader if for_weight_update else self.test_loader

            for batch_idx, (data, target) in enumerate(data_loader):
                try:
                    loss = self.network.get_loss(data, target, self.class_weights, train=False)
                    output = self.network(data)
                except RuntimeError:  # The batch is "not complete"
                    break

                target = target.type(torch.LongTensor).to(DEVICE)

                if torch.cuda.is_available():
                    acc = torch.sum(torch.argmax(output, dim=1) == target).cuda()  # = 0
                else:
                    acc = torch.sum(torch.argmax(output, dim=1) == target).cpu()  # = 0

                loss_test += loss
                self.eval_dic["nb_correct"] += acc
                self.eval_dic["nb_labels"] += len(target)

        if not for_weight_update:
            acc = self.print_eval_model(loss_test)
            self.losses_test["Pretrained Model"].append(round(float(loss_test), 2))
            self.f1_test["Pretrained Model"].append(acc)
            return round(float(loss_test), 2), acc

    '''---------------------- print_eval_model ----------------------------------
     This function prints the different current values of the accuracy, the r
     recalls (related to both pos and neg classes), the f1-measure and the loss
     OUT: the avg of the f1 measures related to each classes 
     ---------------------------------------------------------------------------'''

    def print_eval_model(self, loss_test):
        acc = 100. * self.eval_dic["nb_correct"] / self.eval_dic["nb_labels"]
        nb_test = len(self.test_loader)
        loss_test = loss_test / nb_test  # avg of the loss

        print(" \n------------------------------------------------------------------ ")
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(self.eval_dic["nb_correct"],
                                                                    self.eval_dic["nb_labels"], acc, loss_test))
        print("Baseline: " + str(1/self.eval_dic["nb_labels"]))
        print(" ------------------------------------------------------------------\n ")

        return acc

    '''------------------------- get_optimizer -------------------------------- '''

    def get_optimizer(self, opt_type="Adam"):
        if opt_type == "Adam":
            return optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        elif opt_type == "SGD":
            return optim.SGD(self.network.parameters(), lr=self.lr, momentum=MOMENTUM)
        elif opt_type == "Adagrad":
            return optim.Adagrad(self.network.parameters(), lr=self.lr, weight_decay=self.wd)

    ''' ------------------------------ 
            reset_eval 
    -------------------------------- '''

    def reset_eval(self):
        for metric, value in self.eval_dic.items():
            self.eval_dic[metric] = 0

    ''' ------------------------------ 
            save_model 
    -------------------------------- '''

    def save_model(self, name_model, testing_set):
        try:
            torch.save(self.network, name_model)
        except FileNotFoundError:
            os.mkdir(name_model.split("/")[0])
            torch.save(self.network, name_model)

        with open(name_model.split(".pt")[0] + '_testdata.pkl', 'wb') as output:
            pickle.dump(testing_set, output, pickle.HIGHEST_PROTOCOL)
        print("Model is saved!")

    ''' ------------------------------ 
            visualization 
    -------------------------------- '''

    def visualization(self, num_epoch, used_db, batch_size, opt_type):

        name_fig = "graphs/ds" + used_db + "_" + str(num_epoch) + "_" + str(batch_size) \
                   + "_" + self.loss_type + "_arch" + TYPE_ARCH + "_opti" + opt_type
        visualization_train(range(0, num_epoch, int(round(num_epoch / 5))), self.losses_train,
                            save_name=name_fig + "_train.png")

        visualization_test(self.losses_test, self.f1_test, save_name=name_fig + "_test")

        if self.loss_type[:len("cross_entropy")] == "cross_entropy":
            self.network.visualize_last_output(next(iter(self.test_loader))[0], name_fig + "outputVis")


if __name__ == '__main__':
    pass
