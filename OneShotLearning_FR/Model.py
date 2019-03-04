import os
import torch
import pickle
from NeuralNetwork import Tripletnet, ContrastiveLoss, SoftMax_Net, AutoEncoder_Net, TYPE_ARCH, Classif_Net
from Visualization import visualization_validation, visualization_train
from Dataprocessing import Face_DS
from torch import nn
from torch import optim

#########################################
#       GLOBAL VARIABLES                #
#########################################


MARGIN = 2.0
MOMENTUM = 0.9
GAMMA = 0.1 # for the lr_scheduler - default value 0.1
N_TEST_IMG = 5
PT_BS = 32  # Batch size for pretraining
PT_NUM_EPOCHS = 200
DEVICE_ID = 2
ROUND_DEC = 5

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The code is running on " + DEVICE)
#DEVICE = torch.cuda.device(DEVICE_ID) if torch.cuda.is_available() else 'cpu
#os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % deviceid
print_info_cuda = False
if DEVICE.type == 'cuda' and print_info_cuda:
    # ================================================
    deviceid = 2
    total, used = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')[deviceid].split(',')
    total = int(total)
    used = int(used)

    print(deviceid, 'Total GPU mem:', total, 'used:', used)

    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')
    print("Visible devices: " + str(visible_devices))
    print_info_cuda = True
    print("count " + str(torch.cuda.device_count()))
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    #print(torch.cuda.get_device_name(2))


# Loss combination not considered here


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
        self.loss_type = "None"
        self.is_classifier = False

        self.lr = 0.00001
        self.wd = 0.001
        self.optimizer = None
        self.class_weights = [1, 1]
        self.weighted_classes = True

        if train_param is not None:
            self.set_for_training(train_param, embedding_net, nb_classes)

        self.eval_dic = {"nb_correct": 0, "nb_labels": 0, "recall_pos": 0, "recall_neg": 0, "f1_pos": 0, "f1_neg": 0}

        # For Visualization
        self.losses_validation = {"Pretrained Model": [], "Non-pretrained Model": []}
        self.f1_validation = {"Pretrained Model": [], "Non-pretrained Model": [], "On Training Set": []}
        self.losses_train = []

    '''---------------------- set_for_training --------------------------------- '''

    def set_for_training(self, train_param, embedding_net, nb_classes):

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
        elif train_param["loss_type"] == "ce_classif":
            self.network = Classif_Net(nb_classes=nb_classes)
            self.is_classifier=True
        else:
            print("ERR: Mismatch with loss type")
            raise Exception

        self.loss_type = train_param["loss_type"]

        # ----------------- Optimizer Definition -------------
        try:
            self.lr = train_param["hyper_par"]["lr"]
            self.wd = train_param["hyper_par"]["wd"]
        except KeyError:
            pass  # We keep the default setting

        try:
            self.scheduler, self.optimizer = self.get_optimizer(train_param["hyper_par"])
        except KeyError:
            print("ERR: Key error in optimizer setting")
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
        except IOError: #FileNotFoundError
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

    def train_nonpretrained(self, num_epochs, hyp_par):

        train_param = {"train_loader": self.train_loader, "loss_type": self.loss_type, "hyper_par": hyp_par}
        model_comp = Model(train_param, validation_loader=self.validation_loader, test_loader=self.test_loader)

        for epoch in range(num_epochs):
            print("-------------- Model that was not pretrained ------------------")
            model_comp.train(epoch)
            loss_notPret, f1_notPret = model_comp.prediction()
            self.losses_validation["Non-pretrained Model"].append(loss_notPret)
            self.f1_validation["Non-pretrained Model"].append(f1_notPret)

    '''---------------------------- train --------------------------------
     This function trains the network attached to the model  
     -----------------------------------------------------------------------'''

    def train(self, epoch, autoencoder=False, with_epoch_opt=False):

        # "Active Learning": Overfitting Detection
        if with_epoch_opt and self.is_overfitting():
            return

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

        if self.scheduler is not None: self.scheduler.step()

        if not autoencoder and not self.is_classifier: self.update_weights()
        self.losses_train.append(loss_list)

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
                try:
                    loss = self.network.get_loss(data, target, self.class_weights, train=False)
                    target = target.type(torch.LongTensor).to(DEVICE)

                    # ----------- Case 1: Classification ----------------
                    if self.is_classifier:
                        output = self.network(data)
                        target = target.type(torch.LongTensor).to(DEVICE)

                        if torch.cuda.is_available():
                            acc = torch.sum(torch.argmax(output, dim=1) == target).cuda()  # = 0
                        else:
                            acc = torch.sum(torch.argmax(output, dim=1) == target).cpu()  # = 0

                        acc_loss += loss
                        self.eval_dic["nb_correct"] += acc
                        self.eval_dic["nb_labels"] += len(target)

                    # ----------- Case 2: Siamese Network ---------------
                    else:
                        out_pos, out_neg = self.network(data)
                        target = target.type(torch.LongTensor).to(DEVICE)
                        tar_pos = torch.squeeze(target[:, 0])  # = Only 0 here
                        tar_neg = torch.squeeze(target[:, 1])  # = Only 1 here

                        if torch.cuda.is_available():
                            acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cuda()  # = 0
                            acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cuda()  # = 1
                        else:
                            acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cpu()  # = 0
                            acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cpu()  # = 1

                        acc_loss += loss
                        if not on_train: self.get_evaluation(acc_pos, acc_neg, len(tar_pos), len(tar_neg))

                except RuntimeError:  # The batch is "not complete"
                    break

        # ----------- Evaluation on the validation set (over epochs) ---------------
        if not on_train and validation:
            eval_measure = self.print_eval_model(acc_loss)
            self.losses_validation["Pretrained Model"].append(round(float(acc_loss), ROUND_DEC))
            self.f1_validation["Pretrained Model"].append(eval_measure)
            return round(float(acc_loss), ROUND_DEC), eval_measure

        # ----------- Evaluation on the test set ---------------
        elif not on_train:
            return self.print_eval_model(acc_loss, loader="Test")

        # ----------- Evaluation on the train set ---------------
        else:
            f1_neg_avg = self.eval_dic["f1_neg"] / len(self.train_loader)
            f1_pos_avg = self.eval_dic["f1_pos"] / len(self.train_loader)
            self.f1_validation["On Training Set"].append((f1_neg_avg+f1_pos_avg)/2)
            return f1_pos_avg, f1_neg_avg

    '''---------------------------- update_weights ------------------------------
     This function updates the class weights considering the current f1 measures
     ---------------------------------------------------------------------------'''

    def update_weights(self):
        f1_pos_avg, f1_neg_avg = self.prediction(on_train=True)  # To get f1_measures
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

    def print_eval_model(self, loss_eval, loader="Validation"):
        acc = 100. * self.eval_dic["nb_correct"] / self.eval_dic["nb_labels"]
        nb_eval = len(self.validation_loader)
        loss_eval = loss_eval / nb_eval  # avg of the loss

        print(" \n---------------------------" + loader + "--------------------------------- ")
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(self.eval_dic["nb_correct"],
                                                                    self.eval_dic["nb_labels"], acc, loss_eval))
        if self.is_classifier:
            print("Baseline: " + str(1 / self.eval_dic["nb_labels"]))
            print(" ------------------------------------------------------------------\n ")
            return acc
        else:
            print("Recall Pos is: " + str(100. * self.eval_dic["recall_pos"] / nb_eval) +
                  "   Recall Neg is: " + str(round(100. * self.eval_dic["recall_neg"] / nb_eval, ROUND_DEC)))
            print("f1 Pos is: " + str(100. * self.eval_dic["f1_pos"] / nb_eval) +
                  "          f1 Neg is: " + str(round(100. * self.eval_dic["f1_neg"] / nb_eval, ROUND_DEC)))
            print(" ------------------------------------------------------------------\n ")
            return round(100. * 0.5 * (self.eval_dic["f1_neg"] + self.eval_dic["f1_pos"]) / nb_eval, ROUND_DEC)

    '''------------------------- get_optimizer -------------------------------- '''

    def get_optimizer(self, hyper_par):

        # ----------------------------------
        #       Optimizer Setting
        # ----------------------------------

        optimizer = None
        if hyper_par["opt_type"] == "Adam":
            optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        elif hyper_par["opt_type"] == "SGD":
            optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=MOMENTUM)
        elif hyper_par["opt_type"] == "Adagrad":
            optimizer = optim.Adagrad(self.network.parameters(), lr=self.lr, weight_decay=self.wd)

        # ----------------------------------
        #  Learning Rate Scheduler Setting
        # ----------------------------------

        if hyper_par["lr_scheduler"] is None:
            return None, optimizer
        else:
            if hyper_par["lr_scheduler"] == "ExponentialLR":
                print("optim: " + str(optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, last_epoch=hyper_par["num_epoch"])))
                return optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, last_epoch=hyper_par["num_epoch"]), optimizer
            elif hyper_par["lr_scheduler"] == "StepLR":
                step_size = hyper_par["num_epoch"] / 5
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=GAMMA) #, last_epoch=hyper_par["num_epoch"])
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

        return

    ''' ------------------------------ 
            reset_eval 
    -------------------------------- '''

    def reset_eval(self):
        for metric, value in self.eval_dic.items():
            self.eval_dic[metric] = 0

    ''' ------------------------------ 
            save_model 
    -------------------------------- '''

    def save_model(self, name_model):
        try:
            torch.save(self.network, name_model)
        except IOError: #FileNotFoundError
            os.mkdir(name_model.split("/")[0])
            torch.save(self.network, name_model)

        #with open(name_model.split(".pt")[0] + '_testdata.pkl', 'wb') as output:
        #   pickle.dump(testing_set, output, pickle.HIGHEST_PROTOCOL)
        print("Model is saved!")

    ''' ------------------------------ 
            visualization 
    -------------------------------- '''

    def visualization(self, num_epoch, db, batch_size, opt_type):

        name_fig = "graphs/ds" + db + "_" + str(num_epoch) + "_" + str(batch_size) \
                   + "_" + self.loss_type + "_arch" + TYPE_ARCH + "_opti" + opt_type

        visualization_train(range(0, num_epoch, int(round(num_epoch / 5))), self.losses_train,
                            save_name=name_fig + "_train.png")

        visualization_validation(self.losses_validation, self.f1_validation, save_name=name_fig + "_valid")

        if self.loss_type[:len("cross_entropy")] == "cross_entropy":
            self.network.visualize_last_output(next(iter(self.validation_loader))[0], name_fig + "outputVis")

    ''' ------------------------- is_overfitting ---------------------------------- 
    This function detects overfitting from:
        - the evolution of the loss over time.
        - the difference in f1 measure on the training and the validation sets        
    ----------------------------------------------------------------------------- '''

    def is_overfitting(self):
        return False
