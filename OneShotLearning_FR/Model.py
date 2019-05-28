import os
import torch
import math
import numpy as np
from NeuralNetwork import Triplet_Net, ContrastiveLoss, SoftMax_Net, AutoEncoder_Net, TYPE_ARCH, Classif_Net
from Visualization import visualization_validation, visualization_train
from Dataprocessing import from_zip_to_data, Face_DS, FROM_ROOT
from torch import nn
from torch import optim

#########################################
#       GLOBAL VARIABLES                #
#########################################

ENCODER_DIR = FROM_ROOT + "encoders/"
MOMENTUM = 0.9
GAMMA = 0.1  # for the lr_scheduler - default value 0.1
N_TEST_IMG = 5
PT_BS = 32  # Batch size for pretraining

PT_NUM_EPOCHS = 180
AUTOENCODER_LR = 0.001
RETRAIN_AUTOENCODER = None
EP_SAVE = 30
ROUND_DEC = 5

STOP_TOLERANCE_EPOCH = 35
MIN_AVG_F1 = 65
TOLERANCE_MAX_SAME_F1 = 15
TOL_OVERFITTING = 15  # 10% of difference
MAX_NB_BATCH_PRED = 100
MIN_FOR_SAVE = 70

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
#                    CLASS: Model
#  ================================================================

class Model:
    def __init__(self, train_param=None, train_loader=None, validation_loader=None, test_loader=None,
                 embedding_net=None, network=None, nb_classes=None):

        # Data
        self.train_loader = train_loader
        self.validation_loader = validation_loader  # No need if autoencoder training
        self.test_loader = test_loader

        # Default Initialization
        self.network = network
        self.nb_classes = nb_classes if nb_classes is None or 2 < nb_classes else None  # If None => no classification

        self.class_weights = [1, 1]
        self.weighted_classes = train_param["weighted_class"]

        if train_param is not None:
            is_autoencoder = train_param["loss_type"] is None
            # ----------------- Network Definition (depending on the loss) -------------
            if train_param["loss_type"] == "triplet_loss":
                self.network = Triplet_Net(embedding_net)
            elif train_param["loss_type"] == "triplet_loss_cos":
                self.network = Triplet_Net(embedding_net, metric="Cosine")
            elif train_param["loss_type"] == "triplet_distdif_loss":
                self.network = Triplet_Net(embedding_net, with_dist_loss=True)
            elif train_param["loss_type"] == "triplet_distdif_loss_cos":
                self.network = Triplet_Net(embedding_net, with_dist_loss=True, metric="Cosine")
            elif train_param["loss_type"] == "triplet_and_ce":
                self.network = Triplet_Net(embedding_net, with_ce_loss=True)
            elif train_param["loss_type"] == "triplet_distdif_ce_loss":
                self.network = Triplet_Net(embedding_net, with_dist_loss=True, with_ce_loss=True)
            elif train_param["loss_type"] == "constrastive_loss":
                self.network = ContrastiveLoss(embedding_net)
            elif is_autoencoder:
                self.network = AutoEncoder_Net(embedding_net)
            elif train_param["loss_type"] == "cross_entropy":
                self.network = SoftMax_Net(embedding_net)
            elif train_param["loss_type"] == "ce_classif":
                self.network = Classif_Net(embedding_net, nb_classes=nb_classes)

            else:
                print("ERR: Mismatch with loss type")
                raise Exception

            self.loss_type = train_param["loss_type"]
            self.scheduler, self.optimizer = self.get_optimizer(train_param["hyper_par"], is_autoencoder=is_autoencoder)

            # ----------------- Class Weighting Use -------------
            try:
                self.weighted_classes = train_param["weighted_class"]
            except KeyError:
                pass  # We keep the default setting
        else:
            pass
            # self.loss_type = "None"

        # For Visualization
        self.eval_dic = {"nb_correct": 0, "nb_labels": 0, "recall_pos": 0,
                         "recall_neg": 0, "f1_pos": 0, "f1_neg": 0, "prec_pos": 0, "prec_neg": 0, "nb_eval": 0}

        self.losses_train = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.losses_validation = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.f1_validation = {"Pretrained Model": [], "Non-pretrained Model": [], "On Training Set": []}
        self.acc_validation = {"Pretrained Model": [], "Non-pretrained Model": [], "On Training Set": []}
        self.f1_detail = {"Pretrained Model": 0, "Non-pretrained Model": 0}
        self.pos_recall = {"Pretrained Model": 0, "Non-pretrained Model": 0}

        self.f1_test = {}

        self.train_f1 = 0
        self.do_act_learning = True

    '''------------------------ pretraining ----------------------------------------------
       The function trains an autoencoder based on given training data 
       IN: Face_DS_train: Face_DS objet whose training data is a list of 
           face images represented through tensors 
           autocoder: an autocoder characterized by an encoder and a decoder 
    ------------------------------------------------------------------------------------ '''

    def pretraining(self, Face_DS_train, hyper_par, num_epochs=PT_NUM_EPOCHS, batch_size=PT_BS):

        name_trained_net = ENCODER_DIR + "encoder_al_" + TYPE_ARCH + "_ep" + str(num_epochs) + ".pt"
        try:
            if RETRAIN_AUTOENCODER is not None:
                self.network.embedding_net.load_state_dict(torch.load(RETRAIN_AUTOENCODER))
                name_trained_net = RETRAIN_AUTOENCODER.split(".")[0] + "retrained" + str(num_epochs) + "pt"
                print("The encoder has been loaded and will be retrained!\n")
                raise FileNotFoundError
            else:
                self.network.embedding_net.load_state_dict(torch.load(name_trained_net))
                print("The encoder has been loaded!\n")
        except FileNotFoundError:
            train_data = Face_DS_train.to_single(Face_DS_train) if self.nb_classes is None else Face_DS_train
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            train_param = {"loss_type": None, "hyper_par": hyper_par, "weighted_class": False}
            autoencoder = Model(train_param=train_param, embedding_net=self.network.embedding_net,
                                train_loader=train_loader)

            print(" ------------ Train as Autoencoder ----------------- ")
            for epoch in range(num_epochs):
                autoencoder.train(epoch, autoencoder=True)
                if epoch != 0 and epoch % EP_SAVE == 0:
                    torch.save(autoencoder.network.state_dict(), ENCODER_DIR + "auto_{0:03d}.pwf".format(epoch))
                    autoencoder.network.visualize_dec(epoch=epoch)

            torch.save(self.network.embedding_net.state_dict(), name_trained_net)
            print("The encoder has been saved as " + name_trained_net + "!\n")
            autoencoder.network.visualize_dec()

    '''------------------ train_nonpretrained -------------------------------------
       The function trains a neural network (so that it's performance can be 
       compared to the ones of a NN that was pretrained) 
    -------------------------------------------------------------------------------- '''

    def train_nonpretrained(self, num_epochs, hyp_par, save=None, extra_source=None):

        train_param = {"train_loader": self.train_loader, "loss_type": self.loss_type, "hyper_par": hyp_par,
                       "weighted_class": self.weighted_classes}
        model_comp = Model(train_param, validation_loader=self.validation_loader, test_loader=self.test_loader,
                           train_loader=self.train_loader)

        for epoch in range(num_epochs):
            print("------ Not pretrained Model with arch " + TYPE_ARCH + " and loss " + self.loss_type + " -----------")
            model_comp.train(epoch)
            loss_notPret, f1_notPret, accuracy = model_comp.prediction()

            self.losses_validation["Non-pretrained Model"].append(loss_notPret)
            self.f1_validation["Non-pretrained Model"].append(f1_notPret)
            self.acc_validation["Non-pretrained Model"].append(accuracy)
            self.losses_train["Non-pretrained Model"].append(model_comp.losses_train["Pretrained Model"][-1])
            self.f1_detail["Non-pretrained Model"] = model_comp.f1_detail["Pretrained Model"]
            self.pos_recall["Non-pretrained Model"] = model_comp.pos_recall["Pretrained Model"]

            model_comp.f1_validation["Non-pretrained Model"].append(f1_notPret)
            # model_comp.active_learning(more_data_name=extra_source, mode="Non-pretrained Model")

            if should_break(self.acc_validation["Non-pretrained Model"], epoch):
                break

        if save is not None and MIN_FOR_SAVE < self.acc_validation["Non-pretrained Model"][-1]:
            name_model = save.split("_pre")[0] + "_nonpretrained.pt"
            try:
                torch.save(model_comp.network, name_model)
            except IOError:  # FileNotFoundError
                os.mkdir(name_model.split("/")[0])
                torch.save(model_comp.network, name_model)
            print("Model not pretrained is saved!")

        if self.loss_type != "ce_classif":
            model_comp.prediction(validation=False)
            self.f1_test["Non-pretrained Model"] = model_comp.f1_test["Pretrained Model"]
        else:
            self.f1_test["Non-pretrained Model"] = "None"

    '''---------------------------- train --------------------------------
     This function trains the network attached to the model  
     -----------------------------------------------------------------------'''

    def train(self, epoch, autoencoder=False):

        self.network.train()
        loss_list = []
        # ------- Go through each batch of the train_loader -------
        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()  # clear gradients for this training step

            try:
                if autoencoder:
                    # ----------- CASE 1: Autoencoder Training -----------
                    data = data.to(DEVICE)  # torch.unsqueeze(data, 0)
                    encoded, decoded = self.network(data)
                    loss = nn.MSELoss()(decoded, data)  # mean square error

                else:
                    # ----------- CASE 2: Image Differentiation Training -----------
                    for i in range(len(data)):  # List of 3 tensors
                        data[i] = data[i].to(DEVICE)
                    loss = self.network.get_loss(data, target, self.class_weights)

            except IOError:  # The batch is "not complete"
                print("ERR: An IO error occured in train! ")
                break

            # -----------------------
            #   Backpropagation
            # -----------------------
            a = torch.tensor(list(self.network.embedding_net.parameters())[0].data) if not autoencoder \
                else torch.tensor(list(self.network.encoder.parameters())[0].data)

            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            b = list(self.network.embedding_net.parameters())[0].data if not autoencoder \
                else list(self.network.encoder.parameters())[0].data

            if torch.equal(a, b):
                print("!! WARNING: The weights of the model were not updated!")

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

        if self.scheduler is not None: self.scheduler.step()

        if not autoencoder and self.nb_classes is None and self.weighted_classes: self.update_weights()
        self.losses_train["Pretrained Model"].append(loss_list)

    '''---------------------------- prediction --------------------------------
     This function tests the given model 
     OUT: loss: current loss once evaluated on the set
          f1-measure: current f1-measure once evaluated on the testing set
     -----------------------------------------------------------------------'''

    def prediction(self, on_train=False, validation=True):
        self.network.eval()
        self.reset_eval()

        with torch.no_grad():
            acc_loss = 0

            if on_train:
                data_loader = self.train_loader
            elif validation:
                data_loader = self.validation_loader
            else:
                data_loader = self.test_loader

            for batch_idx, (data, target) in enumerate(data_loader):
                if MAX_NB_BATCH_PRED < batch_idx:
                    break
                try:
                    loss = self.network.get_loss(data, target, self.class_weights, train=False)
                    # target = target.type(torch.LongTensor).to(DEVICE)

                    # ----------- Case 1: Classification ----------------
                    if self.nb_classes is not None:
                        output = self.network(data, target)
                        # target = target.type(torch.LongTensor).to(DEVICE)

                        if torch.cuda.is_available():
                            acc = torch.sum(torch.argmax(output, dim=1) == target).cuda()  # = 0
                        else:
                            acc = torch.sum(torch.argmax(output, dim=1) == target).cpu()  # = 0

                        acc_loss += loss  # Accumulator of losses
                        self.eval_dic["nb_correct"] += acc
                        self.eval_dic["nb_labels"] += len(target)

                    # ----------- Case 2: Siamese Network ---------------
                    else:
                        out_pos, out_neg = self.network(data)
                        target = target.type(torch.LongTensor).to(DEVICE)  # !! Important
                        tar_pos = torch.squeeze(target[:, 0])  # = Only 0 here
                        tar_neg = torch.squeeze(target[:, 1])  # = Only 1 here

                        if torch.cuda.is_available():
                            acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cuda()  # = 0
                            acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cuda()  # = 1
                        else:
                            acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cpu()  # = 0
                            acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cpu()  # = 1

                        acc_loss += loss

                        # if (acc_pos + acc_neg) == len(tar_pos) + len(tar_neg):
                        # print("TODELETE:  STRANGE HERE with tp " + str(acc_pos) + " tn " + str(acc_neg))
                        # print("TODELETE: Outputs are " + str(out_pos) + " and " + str(out_neg))

                        self.get_evaluation(acc_pos, acc_neg, len(tar_pos), len(tar_neg))

                except RuntimeError:  # RuntimeError:  # The batch is "not complete"
                    print("ERR: A runtime error occured in prediction!")
                    break

        f1_neg_avg = self.eval_dic["f1_neg"] / self.eval_dic["nb_eval"]
        f1_pos_avg = self.eval_dic["f1_pos"] / self.eval_dic["nb_eval"]

        # ----------- Evaluation on the validation set (over epochs) ---------------
        if not on_train and validation:
            eval_measure, accuracy = self.print_eval_model(acc_loss)
            self.losses_validation["Pretrained Model"].append(round(float(acc_loss), ROUND_DEC))
            self.f1_validation["Pretrained Model"].append(eval_measure)
            self.acc_validation["Pretrained Model"].append(accuracy)
            self.f1_detail["Pretrained Model"] = (f1_pos_avg, f1_neg_avg)

            return round(float(acc_loss), ROUND_DEC), eval_measure, accuracy

        # ----------- Evaluation on the test set ---------------
        elif not on_train:
            self.f1_test["Pretrained Model"] = self.print_eval_model(acc_loss, loader="Test")

        # ----------- Evaluation on the train set ---------------
        else:
            print("The f1 score evaluated on the training set is " + str((f1_neg_avg + f1_pos_avg) / 2))
            self.f1_validation["On Training Set"].append(round(100. * (f1_neg_avg + f1_pos_avg) / 2))
            return f1_pos_avg, f1_neg_avg

    '''---------------------------- update_weights ------------------------------
     This function updates the class weights considering the current f1 measures
     ---------------------------------------------------------------------------'''

    def update_weights(self):
        print("Loss Weights are updated")
        f1_pos_avg, f1_neg_avg = self.prediction(on_train=True)  # To get f1_measures
        self.train_f1 = (f1_pos_avg + f1_neg_avg) / 2
        print("f1 score of train: " + str(f1_pos_avg) + " and " + str(f1_neg_avg))

        diff = abs(f1_neg_avg - f1_pos_avg)
        # If diff = 0, then weight = 1 (no change)
        if f1_pos_avg < f1_neg_avg:
            # weight_neg -= diff
            self.class_weights[0] += diff
        else:
            # weight_pos -= diff
            self.class_weights[1] += diff

        print("Weights are " + str(self.class_weights[0]) + " and " + str(self.class_weights[1]))

    '''---------------------------------- active_learning ----------------------------------------------
     This function detects overfitting and potentially sets the train_loader 
     to new training data if there's any that is available
     IN: more_data: name of zip file where to take new data 
     OUT: False if the training should stop because of overfitting and lacks of new training data 
     ---------------------------------------------------------------------------------------------------'''

    def active_learning(self, more_data_name=None, mode="Pretrained Model", batch_size=32, nb_people=200):

        if not self.do_act_learning:
            return
        # -------- Compute the f1 measure from the training data --------------------------
        if not self.weighted_classes:
            f1_pos_avg, f1_neg_avg = self.prediction(on_train=True)  # To get f1_measures
            self.train_f1 = (f1_pos_avg + f1_neg_avg) / 2

        # --------------- Check if overfitting --------------------------

        if self.f1_validation[mode][-1] < self.train_f1 - TOL_OVERFITTING:
            if more_data_name is None:
                self.do_act_learning = False
                print("Detected overfitting!\n")
                return True
            else:
                fileset = from_zip_to_data(False, fname=more_data_name)
                dataset = Face_DS(fileset=fileset, nb_people=nb_people)
                self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                print("The new source of data has been used!\n")
                self.do_act_learning = False
                return False
        else:
            return False

    '''---------------------- print_eval_model ----------------------------------
     This function prints the different current values of the accuracy, the r
     recalls (related to both pos and neg classes), the f1-measure and the loss
     OUT: the avg of the f1 measures related to each classes 
          the accuracy 
     ---------------------------------------------------------------------------'''

    def print_eval_model(self, loss_eval, loader="Validation"):
        acc = 100. * self.eval_dic["nb_correct"] / self.eval_dic["nb_labels"]
        nb_eval = self.eval_dic["nb_eval"]
        loss_eval = loss_eval / nb_eval  # avg of the loss

        print(" \n----------------------- " + loader + " --------------------------------- ")
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(self.eval_dic["nb_correct"],
                                                                    self.eval_dic["nb_labels"], acc, loss_eval))
        if self.nb_classes is not None:
            print("Baseline (i.e accuracy in random case): " + str(100 * (float(1) / self.nb_classes)))
            print("------------------------------------------------------------------\n ")
            return acc
        else:
            print("Recall Pos is: " + str(100. * self.eval_dic["recall_pos"] / nb_eval) +
                  "   Recall Neg is: " + str(round(100. * self.eval_dic["recall_neg"] / nb_eval, ROUND_DEC)))
            print("Precision Pos is: " + str(100. * self.eval_dic["prec_pos"] / nb_eval) +
                  "   Precision Neg is: " + str(round(100. * self.eval_dic["prec_neg"] / nb_eval, ROUND_DEC)))
            print("f1 Pos is: " + str(100. * self.eval_dic["f1_pos"] / nb_eval) +
                  "          f1 Neg is: " + str(round(100. * self.eval_dic["f1_neg"] / nb_eval, ROUND_DEC)))
            print(" ------------------------------------------------------------------\n ")

            return round(100. * 0.5 * (self.eval_dic["f1_neg"] + self.eval_dic["f1_pos"]) / nb_eval,
                         ROUND_DEC), acc.item()

    '''------------------------- get_optimizer -------------------------------- '''

    def get_optimizer(self, hyper_par, is_autoencoder=False):

        # ----------------------------------
        #       Optimizer Setting
        # ----------------------------------

        optimizer = None
        lr = hyper_par["lr"] if not is_autoencoder else AUTOENCODER_LR
        wd = hyper_par["wd"]
        if hyper_par["opt_type"] == "Adam":
            optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        elif hyper_par["opt_type"] == "SGD":
            optimizer = optim.SGD(self.network.parameters(), lr=lr, momentum=MOMENTUM)
        elif hyper_par["opt_type"] == "Adagrad":
            optimizer = optim.Adagrad(self.network.parameters(), lr=lr, weight_decay=wd)

        # ----------------------------------
        #  Learning Rate Scheduler Setting
        # ----------------------------------

        if hyper_par["lr_scheduler"] is None or is_autoencoder:
            return None, optimizer
        else:
            if hyper_par["lr_scheduler"] == "ExponentialLR":
                print("\nOptimizer set: " + str(optim.lr_scheduler.ExponentialLR(optimizer, GAMMA)))
                return optim.lr_scheduler.ExponentialLR(optimizer, GAMMA), optimizer
            elif hyper_par["lr_scheduler"] == "StepLR":
                step_size = math.ceil(hyper_par["num_epoch"] / 5)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size,
                                                      gamma=GAMMA)  # , last_epoch=hyper_par["num_epoch"])
                return scheduler, optimizer
            else:
                print("WARN: No mapping with learning rate scheduler")
                return None, optimizer

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
        self.eval_dic["nb_eval"] += 1
        try:
            rec_pos = float(tp) / p
            prec_pos = float(tp) / (float(tp) + (n - float(tn)))

            self.eval_dic["prec_pos"] += prec_pos
            self.eval_dic["recall_pos"] += rec_pos
            self.eval_dic["f1_pos"] += (2 * rec_pos * prec_pos) / (prec_pos + rec_pos)
        except ZeroDivisionError:  # the tp is 0
            pass

        try:
            rec_neg = float(tn) / n
            prec_neg = float(tn) / (float(tn) + (p - float(tp)))

            self.eval_dic["prec_neg"] += prec_neg
            self.eval_dic["recall_neg"] += rec_neg
            self.eval_dic["f1_neg"] += (2 * rec_neg * prec_neg) / (prec_neg + rec_neg)
        except ZeroDivisionError:  # the tn is 0
            pass

        self.eval_dic["nb_correct"] += (tp + tn)
        self.eval_dic["nb_labels"] += p + n

        return

    ''' ------------------------------ 
            reset_eval 
    -------------------------------- '''

    def reset_eval(self):
        try:
            self.pos_recall["Pretrained Model"] = self.eval_dic["recall_pos"] / self.eval_dic["nb_eval"]
        except ZeroDivisionError:
            pass
        for metric, value in self.eval_dic.items():
            self.eval_dic[metric] = 0

    ''' ------------------------------ 
            save_model 
    -------------------------------- '''

    def save_model(self, name_model):
        if self.acc_validation["Pretrained Model"][-1] < MIN_FOR_SAVE:
            print("The last computed accuracy on the validation set is lower than " + str(MIN_FOR_SAVE))
            print("The model wasn't saved!\n")
            return
        try:
            torch.save(self.network, name_model)
        except IOError:  # FileNotFoundError
            os.mkdir(name_model.split("/")[0])
            torch.save(self.network, name_model)

        # with open(name_model.split(".pt")[0] + '_testdata.pkl', 'wb') as output:
        #   pickle.dump(testing_set, output, pickle.HIGHEST_PROTOCOL)
        print("Model is saved as " + name_model + "!")

    ''' ------------------------------ 
            visualization 
    -------------------------------- '''

    def visualization(self, num_epoch, db, size_train):

        name_fig = FROM_ROOT + "graphs/" + db + "_" + str(size_train) + "_" + str(num_epoch) + "_" \
                   + self.loss_type + "_arch" + TYPE_ARCH

        # TOCHANGE int(round(num_epoch / 5))
        visualization_train(num_epoch, self.losses_train, save_name=name_fig + "_train")

        visualization_validation(self.losses_validation, self.f1_validation,
                                 self.acc_validation, num_epoch, save_name=name_fig + "_valid")

        if self.loss_type[:len("cross_entropy")] == "cross_entropy":
            self.network.visualize_last_output(next(iter(self.validation_loader))[0], name_fig + "outputVis")


#########################################
#       GLOBAL VARIABLES                #
#########################################

"""
IN: f1_validation 
    epoch
"""


def should_break(f1_list, epoch, loss_list=None):
    curr_avg_f1 = sum(f1_list) / len(f1_list)

    constant = True
    curr_f1 = f1_list[-1]

    for i in range(TOLERANCE_MAX_SAME_F1):
        if epoch < TOLERANCE_MAX_SAME_F1 or curr_f1 != f1_list[len(f1_list) - 1 - i]:
            constant = False
            break

    if loss_list is not None and TOLERANCE_MAX_SAME_F1 < len(loss_list) + 1:
        avg_last_losses = np.mean(loss_list[len(loss_list) - TOLERANCE_MAX_SAME_F1:])
    else:
        avg_last_losses = 1000  # arbitrary choice so that no interruption

    if (STOP_TOLERANCE_EPOCH < epoch and curr_avg_f1 < MIN_AVG_F1) or constant or avg_last_losses < 0.005:

        print("The f1 measure is bad or constant (during " + str(TOLERANCE_MAX_SAME_F1) + " epochs) => Stop Training")
        return True
    else:
        return False
