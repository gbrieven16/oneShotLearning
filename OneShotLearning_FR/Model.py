import torch
import os
import pickle
from NeuralNetwork import Tripletnet, ContrastiveLoss, SoftMax_Net, AutoEncoder_Net, TYPE_ARCH
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
PT_BS = 32 # Batch size for pretraining
PT_NUM_EPOCHS = 200

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loss combination not considered here


# ================================================================
#                    CLASS: Model
#  ================================================================

class Model:
    def __init__(self, train_param=None, test_loader=None, embedding_net=None, network=None, train_loader=None):

        # Default Initialization
        self.network = network
        self.loss_type = "None"
        self.train_loader = train_loader
        self.lr = 0.001
        self.wd = 0.001
        self.optimizer = None
        self.class_weights = [1, 1]
        self.weighted_classes = True

        if train_param is not None:
            self.set_for_training(train_param, embedding_net)

        self.test_loader = test_loader  # No need if autoencoder training

        self.eval_dic = {"nb_correct": 0, "nb_labels": 0, "recall_pos": 0, "recall_neg": 0, "f1_pos": 0, "f1_neg": 0}

        # For Visualization
        self.losses_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.f1_test = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.losses_train = []

    def set_for_training(self, train_param, embedding_net):

        # ----------------- Network Definition -------------
        if train_param["loss_type"] == "triplet_loss":
            self.network = Tripletnet()
        elif train_param["loss_type"] == "constrastive_loss":
            self.network = ContrastiveLoss()
        elif train_param["loss_type"] is None:
            self.network = AutoEncoder_Net(embedding_net)
        elif train_param["loss_type"][:len("cross_entropy")] == "cross_entropy":
            self.network = SoftMax_Net() if len(train_param["loss_type"]) == len("cross_entropy") else SoftMax_Net(
                with_center_loss=True)

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
            pass   # We keep the default setting

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

            autoencoder = Model(train_param, embedding_net=self.network.embedding_net, train_loader=train_loader)

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
        model_comp = Model(train_param, test_loader=self.test_loader)

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

        if not autoencoder: self.update_weights()
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
                print("MODEL: TEST: One batch is considered...")
                try:
                    loss = self.network.get_loss(data, target, self.class_weights, train=False)
                    out_pos, out_neg = self.network(data)
                except RuntimeError:  # The batch is "not complete"
                    break

                print("MODEL: TEST: After get loss...")
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
                self.get_evaluation(acc_pos, acc_neg, len(tar_pos), len(tar_neg))
                print("MODEL: TEST: End Evaluation \n")

        if not for_weight_update:
            f1_measure = self.print_eval_model(loss_test)
            self.losses_test["Pretrained Model"].append(round(float(loss_test), 2))
            self.f1_test["Pretrained Model"].append(f1_measure)
            return round(float(loss_test), 2), f1_measure

    '''---------------------------- update_weights ------------------------------
     This function updates the class weights considering the current f1 measures
     ---------------------------------------------------------------------------'''

    def update_weights(self):
        self.test(for_weight_update=True) # To get f1_measures
        f1_neg_avg = self.eval_dic["f1_neg"] / len(self.train_loader)
        f1_pos_avg = self.eval_dic["f1_pos"] / len(self.train_loader)
        print("f1 score of train: " + str(f1_pos_avg) + " and " + str(f1_neg_avg))

        if self.weighted_classes:
            diff = abs(f1_neg_avg - f1_pos_avg)
            # If diff = 0, then weight = 1 (no change)
            if f1_pos_avg < f1_neg_avg:
                # weight_neg -= diff
                self.class_weights[0] += diff
            else:
                # weight_pos -= diff
                self.class_weights[1] += diff
        print("Weights are " + str(self.class_weights[0]) + " and " + str(self.class_weights[1]))

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
        print("Recall Pos is: " + str(100. * self.eval_dic["recall_pos"] / nb_test) +
              "   Recall Neg is: " + str(round(100. * self.eval_dic["recall_neg"] / nb_test, 2)))
        print("f1 Pos is: " + str(100. * self.eval_dic["f1_pos"] / nb_test) +
              "          f1 Neg is: " + str(round(100. * self.eval_dic["f1_neg"] / nb_test, 2)))
        print(" ------------------------------------------------------------------\n ")

        return round(100. * 0.5 * (self.eval_dic["f1_neg"] + self.eval_dic["f1_pos"]) / nb_test, 2)

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

    def get_evaluation(self, tp, tn, p, n):

        try:
            rec_pos = float(tp) / p
            self.eval_dic["recall_pos"] += rec_pos
            prec_pos = float(tp) / (float(tp) + (n - float(tn)))
            self.eval_dic["f1_pos"] += (2 * rec_pos * prec_pos) / (prec_pos + rec_pos)

        except ZeroDivisionError:  # the tp is 0
            pass

        try:
            rec_neg = float(tn) / n
            self.eval_dic["recall_neg"] += rec_neg
            prec_neg = float(tn) / (float(tn) + (p - float(tp)))
            self.eval_dic["f1_neg"] += (2 * rec_neg * prec_neg) / (prec_neg + rec_neg)

        except ZeroDivisionError:  # the tn is 0
            pass

        self.eval_dic["nb_correct"] += (tp + tn)
        self.eval_dic["nb_labels"] += p + n

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
