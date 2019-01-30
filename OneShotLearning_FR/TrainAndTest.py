import torch
import torch.nn.functional as f
from NeuralNetwork import Tripletnet, AutoEncoder
from torch import nn
from torch import optim

DIST_THRESHOLD = 0.02

MARGIN = 2.0
MOMENTUM = 0.9
N_TEST_IMG = 5

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''---------------------------- train --------------------------------
 This function trains the model 
 OUT: loss_list (list of losses during the epoch) 
      perc_ts_list (list of the corresponding % of data which is used) 
 -----------------------------------------------------------------------'''


def train(model, device, train_loader, epoch, optimizer, loss_type="cross_entropy", autoencoder=False):
    model.train()
    loss_list = []
    # ------- Go through each batch of the train_loader -------
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # clear gradients for this training step
        loss = None

        try:
            if autoencoder:
                # ----------------------------
                # CASE 1: Autoencoder Training
                # ----------------------------
                data_copy = data  # torch.unsqueeze(data, 0)
                encoded, decoded = model([data])
                loss = nn.MSELoss()(decoded, data_copy)  # mean square error
            else:
                # ----------------------------------------------
                # CASE 2: Image Differentiation Training
                # ----------------------------------------------
                _, _, _, _, loss = compute_loss(data, target, device, model, loss_type, with_output=False)
        except IOError:  # The batch is "not complete"
            print("A runtime error occurred")
            print("Data is: " + str(data))
            print("Target is: " + str(target))
            break

        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # -----------------------
        #   Visualization
        # -------------------------
        if batch_idx % 10 == 0:  # Print the state of the training each 10 batches (i.e each 10*size_batch considered examples)
            if autoencoder:
                batch_size = len(data)
            else:
                batch_size = len(data[0])
            loss_list.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx * batch_size / len(train_loader.dataset),
                loss.item()))  # len(data[0]) = batch_size
    return loss_list


'''---------------------------- test --------------------------------
 This function tests the given model 
 OUT: loss: current loss once evaluated on the testing set
      accuracy: current accuracy once evaluated on the testing set
 -----------------------------------------------------------------------'''


def test(model, device, test_loader, loss_type="cross_entropy"):
    model.eval()

    with torch.no_grad():
        eval_dic = {"nb_correct": 0, "nb_labels": 0, "recall_pos": 0, "recall_neg": 0, "f1_pos": 0, "f1_neg": 0}
        loss_test = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                out_pos, out_neg, tar_pos, tar_neg, loss = compute_loss(data, target, device, model, loss_type)
            except RuntimeError:  # The batch is "not complete"
                break

            if device.type == "cpu":
                acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cpu()  # = 0
                acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cpu()  # = 1
            else:
                acc_pos = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cuda()  # = 0
                acc_neg = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cuda()  # = 1

            loss_test += loss
            get_evaluation(acc_pos, acc_neg, len(tar_pos), len(tar_neg), eval_dic)

    acc = 100. * eval_dic["nb_correct"] / eval_dic["nb_labels"]
    loss_test = loss_test / len(test_loader)  # avg of the loss

    print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(eval_dic["nb_correct"], eval_dic["nb_labels"], acc, loss_test))
    print("Recall Pos is: " + str(100. * eval_dic["recall_pos"] / len(test_loader)))
    print("Recall Neg is: " + str(100. * eval_dic["recall_neg"] / len(test_loader)))
    print("f1 Pos is: " + str(100. * eval_dic["f1_pos"] / len(test_loader)))
    print("f1 Neg is: " + str(100. * eval_dic["f1_neg"] / len(test_loader)))

    return round(float(loss_test), 2), acc


'''------------------------- compute_loss --------------------------------
 This function derives the loss corresponding to the given data, providing
 to the model

 IN: data: list of 3 tensors, each respectively representing the anchor,
           the positive and the negative 
     target: the corresponding pairs of labels ((0, 1) here) 
     device: device to use 
     model: the model we evaluate the loss from 

 OUT: computed output_positive, output_negative
      formatted target 
      loss
 -----------------------------------------------------------------------'''


def compute_loss(data, target, device, model, loss_type, with_output=True):
    loss = 0
    for i in range(len(data)):  # List of 3 tensors
        data[i] = data[i].to(device)

    output_positive = None
    output_negative = None

    target = target.type(torch.LongTensor).to(device)
    target_positive = torch.squeeze(target[:, 0])  # = Only 0 here
    target_negative = torch.squeeze(target[:, 1])  # = Only 1 here

    # -----------------------------
    # CASE 1: Cross Entropy
    # -----------------------------
    if loss_type == "cross_entropy":
        output_positive = model(data[:2])  # 2 elements for each batch because 2 classes (pos, neg)
        output_negative = model(data[0:3:2])
        loss_positive = f.cross_entropy(output_positive, target_positive)
        loss_negative = f.cross_entropy(output_negative, target_negative)
        loss = loss_positive + loss_negative

    # -----------------------------
    # CASE 2: Triplet Loss
    # -----------------------------
    elif loss_type == "triplet_loss":
        criterion = torch.nn.MarginRankingLoss(margin=MARGIN)

        tnet = Tripletnet(model)
        if device == "cuda": tnet.cuda()

        dista, distb, embedded_0, embedded_2, embedded_1 = tnet.forward(data[0], data[2], data[1])
        update_dist_threshold(dista, distb)

        # 1 means, dista should be greater than distb
        target = torch.FloatTensor(dista.size()).fill_(1).to(device)

        loss = criterion(dista, distb, target)

        if with_output:
            output_positive = torch.ones([dista.size()[0], 2], dtype=torch.float64).to(device)
            output_positive[distb + 0.2 * distb <= DIST_THRESHOLD, 1] = 0
            output_positive[distb > DIST_THRESHOLD - 0.2 * distb, 1] = 2

            output_negative = torch.ones([dista.size()[0], 2], dtype=torch.float64).to(device)
            output_negative[DIST_THRESHOLD <= dista, 0] = 0
            output_negative[dista < DIST_THRESHOLD, 0] = 2

    # -----------------------------
    # CASE 3: Contrastive Loss
    # -----------------------------
    elif loss_type == "constrastive_loss":
        pass

    return output_positive, output_negative, target_positive, target_negative, loss


'''---------------------------- oneshot ---------------------------
   The function predicts if 2 images represent the same person  
   IN: model: neural network taking 2 processed images as input and
              outputting a pair of numbers
       data: list of 2 tensors representing 2 images 
   OUT: index of the maximum value in the output
        This index corresponds to the predicted class: 
        0 => same person  &  1 => different people 
------------------------------------------------------------------- '''


def oneshot(model, data):
    model.eval()
    # print("Data given to oneshot: " + str(data))
    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(DEVICE)

        output = model(data)
        if DEVICE == "cpu":
            return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
        else:
            return torch.squeeze(torch.argmax(output, dim=1)).cuda().item()


'''---------------------------------- pretraining -----------------------------------------------
   The function trains an autoencoder based on given training data 
   IN: train_data: Face_DS objet whose training data is a list of 
       face images represented through tensors 
       autocoder: an autocoder characterized by an encoder and a decoder 
       loss_type: loss to use to pretrain the encoder (if it's not "autoencoder")
   OUT: the encoder (after training) 
---------------------------------------------------------------------------------------------- '''


def pretraining(train_data, autocoder, num_epochs=100, batch_size=32, loss_type=None):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = get_optimizer(autocoder)

    if loss_type == "autoencoder":
        print(" ------------ Train as Autoencoder ----------------- ")
        for epoch in range(num_epochs):
            train(autocoder, DEVICE, train_loader, epoch, optimizer, autoencoder=True)
    else:
        print(" ------------ Pretraining with " + loss_type + " ----------------- ")
        for epoch in range(num_epochs):
            train(autocoder.encoder, DEVICE, train_loader, epoch, optimizer, loss_type=loss_type, autoencoder=False)

    return autocoder.encoder  # Returns the pretrained Model


'''------------------ train_nonpretrained -------------------------------------
   The function trains a neural network (so that it's performance can be 
   compared to the ones of a NN that was pretrained) 
-------------------------------------------------------------------------------- '''


def train_nonpretrained(train_loader, test_loader, losses_test, acc_test, num_epochs, loss_type, optimizer_type):
    model_comp = AutoEncoder(device=DEVICE).encoder
    optimizer = get_optimizer(model_comp, optimizer_type)
    # ------- Model Training ---------
    for epoch in range(num_epochs):
        print("-------------- Model that was not pretrained ------------------")
        train(model_comp, DEVICE, train_loader, epoch, optimizer, loss_type=loss_type)
        loss_notPret, acc_notPret = test(model_comp, DEVICE, test_loader, loss_type=loss_type)
        losses_test["Non-pretrained Model"].append(loss_notPret)
        acc_test["Non-pretrained Model"].append(acc_notPret)


'''------------------ get_optimizer ---------------------------------- '''


def get_optimizer(model, opt_type="Adam", learning_rate=0.001, weight_decay=0.001):
    if opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)
    elif opt_type == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


'''----------- update_dist_threshold ------------------------------ 
   This function updates the distance threshold by avg: 
        - the distances separting faces of different people,
        - the distances separting faces of same people,
        - the previous threshold value 
-----------------------------------------------------------------'''


def update_dist_threshold(dista, distb):
    avg_dista = float(sum(dista)) / dista.size()[0]
    avg_distb = float(sum(distb)) / dista.size()[0]
    global DIST_THRESHOLD
    DIST_THRESHOLD = (DIST_THRESHOLD + avg_dista + avg_distb) / 3


'''----------- get_evaluation ------------------------------ 
   This function computes different metrics to evaluate the
   performance some model, based on:
    - tp (True Positives) 
    - tn (True Negatives) 
    - p (Total number of positives) 
    - n (Total number of negatives) 
    IN: eval_dic: accumulator dictionary whose keys are: 
    "nb_correct", "nb_labels", "recall_pos", "recall_neg", "f1_pos", "f1_neg"

------------------------------------------------------------------------'''


def get_evaluation(tp, tn, p, n, eval_dic):
    eval_dic["recall_pos"] += float(tp) / (float(tp) + (p - float(tp)))
    eval_dic["recall_neg"] += float(tn) / (float(tn) + (n - float(tn)))

    prec_pos = float(tp) / (float(tp) + (n - float(tn)))
    prec_neg = float(tn) / (float(tn) + (p - float(tp)))

    eval_dic["f1_pos"] += (2 * eval_dic["recall_pos"] * prec_pos) / (prec_pos + eval_dic["recall_pos"])
    eval_dic["f1_neg"] += (2 * eval_dic["recall_neg"] * prec_neg) / (prec_neg + eval_dic["recall_neg"])

    eval_dic["nb_correct"] += (tp + tn)
    eval_dic["nb_labels"] += p + n


if __name__ == '__main__':
    pass
