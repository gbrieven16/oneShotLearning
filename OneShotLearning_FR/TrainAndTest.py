import torch
import torch.nn.functional as f
from NeuralNetwork import Tripletnet

from Visualization import multi_line_graph, line_graph

LOSS = "cross_entropy"

'''---------------------------- train --------------------------------
 This function trains the model 
 OUT: loss_list (list of losses during the epoch) 
      perc_ts_list (list of the corresponding % of data which is used) 
 -----------------------------------------------------------------------'''


def train(model, device, train_loader, epoch, optimizer, batch_size):
    model.train()
    loss_list = []

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        _, _, _, _, loss = compute_loss(data, target, device, model)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # Print the state of the training each 10 batches (i.e each 10*size_batch considered examples)
            loss_list.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx * batch_size / len(train_loader.dataset), loss.item()))
    return loss_list


'''---------------------------- test --------------------------------
 This function tests the given model 
 OUT: loss: current loss once evaluated on the testing set
      accuracy: current accuracy once evaluated on the testing set
 -----------------------------------------------------------------------'''


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        nb_labels = 0
        loss_test = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            out_pos, out_neg, tar_pos, tar_neg, loss = compute_loss(data, target, device, model)
            if device.type == "cpu":
                accurate_labels_positive = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cpu()  # = 0
                accurate_labels_negative = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cpu()  # = 1
            else:
                accurate_labels_positive = torch.sum(torch.argmax(out_pos, dim=1) == tar_pos).cuda()  # = 0
                accurate_labels_negative = torch.sum(torch.argmax(out_neg, dim=1) == tar_neg).cuda()  # = 1

            loss_test += loss
            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            nb_labels = nb_labels + len(tar_pos) + len(tar_neg)

        accuracy = 100. * accurate_labels / nb_labels
        loss_test = loss_test / len(test_loader)  # avg of the loss
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, nb_labels, accuracy, loss_test))

    return loss_test, accuracy


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


def compute_loss(data, target, device, model):
    loss = 0
    for i in range(len(data)):
        data[i] = data[i].to(device)

    output_positive = model(data[:2])  # 2 elements for each batch because 2 classes (pos, neg) (= (1-diff, diff))
    output_negative = model(data[0:3:2])

    target = target.type(torch.LongTensor).to(device)
    target_positive = torch.squeeze(target[:, 0])
    target_negative = torch.squeeze(target[:, 1])

    if LOSS == "cross_entropy":
        loss_positive = f.cross_entropy(output_positive, target_positive)
        loss_negative = f.cross_entropy(output_negative, target_negative)
        loss = loss_positive + loss_negative

    elif LOSS == "triplet_loss":
        criterion = torch.nn.MarginRankingLoss(margin=1.0)

        tnet = Tripletnet(model)
        if device == "cuda": tnet.cuda()

        dista, distb, _, _, _ = tnet.forward(data[0], data[2], data[1])

        # 1 means, dista should be greater than distb
        target = torch.FloatTensor(dista.size()).fill_(1).to(device)

        loss = criterion(dista, distb, target)

    return output_positive, output_negative, target_positive, target_negative, loss


'''---------------------------- oneshot ---------------------------'''


def oneshot(model, device, data):
    model.eval()
    # print("Data given to oneshot: " + str(data))
    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


'''------------------ visualization_train -------------------------------------------
IN: epoch_list: list of specific epochs
    loss_list: list of lists of all the losses during each epoch
--------------------------------------------------------------------------------------'''


def visualization_train(epoch_list, loss_list, save_name=None):
    title = "Evolution of the loss for different epoches"
    perc_train = [x / len(loss_list[0]) for x in range(0, len(loss_list[0]))]
    dictionary = {}
    for i, epoch in enumerate(epoch_list):
        dictionary["epoch " + str(epoch)] = loss_list[epoch]

    multi_line_graph(dictionary, perc_train, title, x_label="percentage of data", y_label="Loss", save_name=save_name)


'''------------------ visualization_test ----------------------------- '''


def visualization_test(loss, acc, save_name=None):
    line_graph(range(0, len(loss), 1), loss, "Loss according to the epochs", x_label="Epoch", y_label="Loss",
               save_name=save_name)
    line_graph(range(0, len(acc), 1), acc, "Accuracy according to the epochs", x_label="Epoch", y_label="Accuracy",
               save_name=save_name)


if __name__ == '__main__':
    loss = [1, 0.2, 0.1, 0.05]
    acc = [0.5, 0.6, 0.7, 0.75]
    visualization_test(loss, acc)

    epoch_list = [0, 2, 4]
    loss_list = [[1, 2, 3], [1, 2, 3], [7, 8, 10], [11, 12, 23], [1, 3, 9]]
    visualization_train(epoch_list, loss_list)
