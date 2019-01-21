import torch
import torch.nn.functional as f
from visualization import multi_line_graph, line_graph

'''---------------------------- train --------------------------------
 This function trains the model 
 OUT: loss_list (list of losses during the epoch) 
      perc_ts_list (list of the corresponding % of data which is used) 
 -----------------------------------------------------------------------'''


def train(model, device, train_loader, epoch, optimizer, batch_size):
    model.train()
    loss_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output_positive = model(data[:2])
        output_negative = model(data[0:3:2])

        target = target.type(torch.LongTensor).to(device)
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])

        loss_positive = f.cross_entropy(output_positive, target_positive)
        loss_negative = f.cross_entropy(output_negative, target_negative)

        loss = loss_positive + loss_negative
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
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output_positive = model(data[:2])
            output_negative = model(data[0:3:2])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = f.cross_entropy(output_positive, target_positive)
            loss_negative = f.cross_entropy(output_negative, target_negative)

            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))

    return loss, accuracy


'''---------------------------- oneshot ---------------------------'''


def oneshot(model, device, data):
    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


'''------------------ visualization_train -------------------------------------------
IN: epoch_list: list of specific epochs
    loss_list: list of lists of all the losses during each epoch
--------------------------------------------------------------------------------------'''


def visualization_train(epoch_list, loss_list):
    title = "Evolution of the loss for different epoches"
    perc_train = [x / len(loss_list[0]) for x in range(0, len(loss_list[0]))]
    dictionary = {}
    for i, epoch in enumerate(epoch_list):
        dictionary["epoch " + str(epoch)] = loss_list[epoch]

    multi_line_graph(dictionary, perc_train, title, x_label="percentage of data", y_label="Loss")


'''------------------ visualization_test ----------------------------- '''


def visualization_test(loss, acc):
    line_graph(range(0, len(loss),1), loss, "Loss according to the epochs", x_label="Epoch", y_label="Loss")
    line_graph(range(0, len(acc),1), acc, "Accuracy according to the epochs", x_label="Epoch", y_label="Accuracy")


if __name__ == '__main__':
    loss = [1, 0.2, 0.1, 0.05]
    acc = [0.5, 0.6, 0.7, 0.75]
    visualization_test(loss, acc)

    epoch_list = [0, 2, 4]
    loss_list = [[1,2, 3],[1,2, 3], [7,8, 10], [11,12, 23], [1,3, 9]]
    visualization_train(epoch_list, loss_list)

