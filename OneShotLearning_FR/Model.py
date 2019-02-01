import torch
import os
import pickle
from NeuralNetwork import Tripletnet, ContrastiveLoss, SoftMax_Net, AutoEncoder, TYPE_ARCH
from Visualization import visualization_test, visualization_train
from torch import nn
from torch import optim

#########################################
#       GLOBAL VARIABLES                #
#########################################

# Parameters to predict if 2 faces represent the same person
WITH_UPDATE_MARG = False

MARGIN = 2.0
MOMENTUM = 0.9
N_TEST_IMG = 5

NUM_EPOCHS_PRETRAINING = 500

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loss combination not considered here
# Main to adapt 


# ================================================================
#                    CLASS: Model
# ================================================================

class Model:
    def __init__(self, train_loader, loss_type, tl=None, hyper_par=None, opt_type="Adam", embedding_net=None, weighted_class=True):
        self.train_loader = train_loader
        self.test_loader = tl

        if loss_type == "triplet_loss":
            self.network = Tripletnet()
        elif loss_type == "constrastive_loss":
            self.network = ContrastiveLoss()
        elif loss_type == "cross_entropy":
            self.network = SoftMax_Net()
        elif embedding_net is not None:
            self.network = AutoEncoder(embedding_net)

        self.loss_type = loss_type
        self.lr = 0.001 if hyper_par is None else hyper_par["lr"]
        self.wd = 0.001 if hyper_par is None else hyper_par["wd"]
        self.optimizer = self.get_optimizer(opt_type=opt_type)
        self.class_weights = [1, 1]
        self.weighted_classes = weighted_class
        self.batch_size = len(self.train_loader)
        self.eval_dic = {"nb_correct": 0, "nb_labels": 0, "recall_pos": 0, "recall_neg": 0, "f1_pos": 0, "f1_neg": 0}
        self.update_marg = False if loss_type == "cross_entropy" else WITH_UPDATE_MARG

        # For Visualization
        self.losses_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.acc_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.losses_train = []

    '''------------------------ pretraining ----------------------------------------------
       The function trains an autoencoder based on given training data 
       IN: train_data: Face_DS objet whose training data is a list of 
           face images represented through tensors 
           autocoder: an autocoder characterized by an encoder and a decoder 
    ------------------------------------------------------------------------------------ '''

    def pretraining(self, train_data, num_epochs=NUM_EPOCHS_PRETRAINING, batch_size=32, loss_type=None):

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        autoencoder = Model(train_loader, loss_type, embedding_net=self.network.embedding_net)

        print(" ------------ Train as Autoencoder ----------------- ")
        for epoch in range(num_epochs):
            autoencoder.train(epoch, autoencoder=True)
        autoencoder.network.visualize_dec()

    '''------------------ train_nonpretrained -------------------------------------
       The function trains a neural network (so that it's performance can be 
       compared to the ones of a NN that was pretrained) 
    -------------------------------------------------------------------------------- '''

    def train_nonpretrained(self, num_epochs, optimizer_type):
        model_comp = Model(self.train_loader, self.loss_type, tl=self.test_loader, opt_type=optimizer_type)

        # ------- Model Training ---------
        for epoch in range(num_epochs):
            print("-------------- Model that was not pretrained ------------------")
            model_comp.train(epoch)
            loss_notPret, acc_notPret = model_comp.test()
            self.losses_test["Non-pretrained Model"].append(loss_notPret)
            self.acc_test["Non-pretrained Model"].append(acc_notPret)

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

            # -----------------------
            #   Visualization
            # -------------------------
            if batch_idx % 10 == 0:
                if autoencoder:
                    batch_size = len(data)
                else:
                    batch_size = len(data[0])
                loss_list.append(loss.item())
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(self.train_loader.dataset),
                           100. * batch_idx * batch_size / len(self.train_loader.dataset),
                    loss.item()))  # len(data[0]) = batch_size

        self.losses_train.append(loss_list)

    '''---------------------------- test --------------------------------
     This function tests the given model 
     OUT: loss: current loss once evaluated on the testing set
          accuracy: current accuracy once evaluated on the testing set
     -----------------------------------------------------------------------'''

    def test(self):
        self.network.eval()
        self.reset_eval()

        with torch.no_grad():
            loss_test = 0

            for batch_idx, (data, target) in enumerate(self.test_loader):
                try:
                    loss = self.network.get_loss(data, target, self.class_weights)
                    out_pos, out_neg = self.network(data)
                except RuntimeError:  # The batch is "not complete"
                    break

                target = target.type(torch.LongTensor).to(DEVICE)
                tar_pos = torch.squeeze(target[:, 0])  # = Only 0 here
                tar_neg = torch.squeeze(target[:, 1])  # = Only 1 here

                if torch.cuda.is_available():
                    acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cuda()  # = 0
                    acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cuda()  # = 1
                else:
                    acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cpu()  # = 0
                    acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cpu()  # = 1

                loss_test += loss
                self.get_evaluation(acc_pos, acc_neg, len(tar_pos), len(tar_neg), self.eval_dic)
                if WITH_UPDATE_MARG: self.network.update_perc_marg_dist(self.eval_dic["f1_pos"],
                                                                        self.eval_dic["f1_neg"])

        acc = self.print_eval_model(loss_test)
        self.update_weights()

        self.losses_test["Pretrained Model"].append(round(float(loss_test), 2))
        self.acc_test["Pretrained Model"].append(acc)
        return round(float(loss_test), 2), acc

    '''---------------------------- update_weights ------------------------------
     This function updates the class weights considering the current f1 measures
     ---------------------------------------------------------------------------'''

    def update_weights(self):
        if self.weighted_classes:
            diff = abs(self.eval_dic["f1_neg"] / self.batch_size - self.eval_dic["f1_pos"] / self.batch_size)
            # If diff = 0, then weight = 1 (no change)
            if self.eval_dic["f1_pos"] / self.batch_size < self.eval_dic["f1_neg"] / self.batch_size:
                # weight_neg -= diff
                self.class_weights[0] += diff
            else:
                # weight_pos -= diff
                self.class_weights[1] += diff

    '''---------------------- print_eval_model ----------------------------------
     This function prints the different current values of the accuracy, the r
     recalls (related to both pos and neg classes), the f1-measure and the loss
     ---------------------------------------------------------------------------'''

    def print_eval_model(self, loss_test):
        acc = 100. * self.eval_dic["nb_correct"] / self.eval_dic["nb_labels"]
        loss_test = loss_test / len(self.test_loader)  # avg of the loss

        print(" \n------------------------------------------------------------------ ")
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(self.eval_dic["nb_correct"],
                                                                    self.eval_dic["nb_labels"], acc, loss_test))
        print("Recall Pos is: " + str(100. * self.eval_dic["recall_pos"] / self.batch_size) +
              "   Recall Neg is: " + str(round(100. * self.eval_dic["recall_neg"] / self.batch_size, 2)))
        print("f1 Pos is: " + str(100. * self.eval_dic["f1_pos"] / self.batch_size) +
              "          f1 Neg is: " + str(round(100. * self.eval_dic["f1_neg"] / self.batch_size, 2)))
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

    '''------------------------- get_evaluation -------------------------------- 
       This function computes different metrics to evaluate the
       performance some model, based on:
        - tp (True Positives) 
        - tn (True Negatives) 
        - p (Total number of positives) 
        - n (Total number of negatives) 
        IN: eval_dic: accumulator dictionary whose keys are: 
        "nb_correct", "nb_labels", "recall_pos", "recall_neg", "f1_pos", "f1_neg"
    ------------------------------------------------------------------------------'''

    def get_evaluation(self, tp, tn, p, n, eval_dic):
        try:
            rec_pos = float(tp) / (float(tp) + (p - float(tp)))
            eval_dic["recall_pos"] += rec_pos
            prec_pos = float(tp) / (float(tp) + (n - float(tn)))
            eval_dic["f1_pos"] += (2 * rec_pos * prec_pos) / (prec_pos + rec_pos)

        except ZeroDivisionError:  # the tp is 0
            pass

        try:
            rec_neg = float(tn) / (float(tn) + (n - float(tn)))
            eval_dic["recall_neg"] += rec_neg
            prec_neg = float(tn) / (float(tn) + (p - float(tp)))
            eval_dic["f1_neg"] += (2 * rec_neg * prec_neg) / (prec_neg + rec_neg)

        except ZeroDivisionError:  # the tn is 0
            pass

        eval_dic["nb_correct"] += (tp + tn)
        eval_dic["nb_labels"] += p + n

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

    def visualization(self, num_epoch, used_db):
        name_fig = "graphs/ds" + used_db + "_" + str(num_epoch) + "_" + str(self.batch_size) \
                   + "_" + self.loss_type + "_arch" + TYPE_ARCH
        visualization_train(range(0, num_epoch, int(round(num_epoch / 5))), self.losses_train,
                            save_name=name_fig + "_train.png")

        visualization_test(self.losses_test, self.acc_test, save_name=name_fig + "_test")
