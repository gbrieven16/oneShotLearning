import platform

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import random
import os
import csv
import numpy as np

# ================================================================
#                       GLOBAL VARIABLES
# ================================================================

FROM_ROOT = "" if platform.system() == "Darwin" else "/home/gbrieven/oneShotLearning/OneShotLearning_FR/"
CSV_NAME = FROM_ROOT + "model_evaluation.csv"
FR_CSV_NAME = FROM_ROOT + "result/fr_model_evaluation.csv"
NB_DATA_GRAPH = 1000
MAX_NB_KEYS = 12
WIDTH_BAR = 0.37

INDEX_BEGIN_GRAPH_LABEL = 0
INDEX_END_GRAPH_LABEL = 9


# ================================================================
#         IMPLEMENTATION OF THE VISUALIZATION TOOLS
# ================================================================


def line_graph(x, y, title, x_label="x", y_label="y", save_name=None):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()
    # --------- Save -----------
    plt.show()
    if save_name is not None:
        try:
            plt.savefig(save_name)
        except FileNotFoundError:
            os.mkdir(save_name.split("/")[0])
            plt.savefig(save_name)
        print("Graph saved as " + save_name)
    plt.close()


"""
IN: dictionary: dictionary where each key is represented by one line
"""


def multi_line_graph(dictionary, x_elements, title, x_label="x", y_label="Score", save_name=None, loc='lower right'):
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # --- We go through all the line to represent ------
    for line in dictionary.keys():

        # ------ Plot the evolution of the value over x elements --------
        try:
            if len(dictionary[line]) < len(x_elements):
                plt.plot(x_elements[:len(dictionary[line])], dictionary[line], label=str(line))
            else:
                plt.plot(x_elements, dictionary[line][:len(x_elements)], label=str(line))
        except ValueError:
            print("Different dimensions in visualisation...\n")

    # --------- Legend -----------------------------
    legend = plt.legend(loc=loc, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # --------- Set the fontsize ------------------
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # --------- Save -----------
    plt.show()
    if save_name is not None:
        try:
            plt.savefig(save_name)
        except FileNotFoundError:
            os.mkdir(save_name.split("/")[0])
            plt.savefig(save_name)
        print("Graph saved as " + save_name)
    plt.close()


# -------------- bar_chart --------------------
# IN: dictionary1 and dictionary2 have the same
#     keys and 1 value in a list
# ---------------------------------------------

def bar_chart(dictionary1, dictionary2, title, dictionary3=None, first_title='Average', second_title='Std',
              third_title="3e title", annotated=True, y_title="Response time (ms)", save_name=None, loc='lower right'):
    if len(dictionary1) == 0:
        print("INFO: In Visualization: Empty dictionaries")
        return None

    fig = plt.figure()
    # plt.grid(True, linewidth=0.5, color='#DCDCDC', linestyle='-')
    # plt.rc('axes', axisbelow=True)
    ax = fig.add_subplot(111)
    ind = np.arange(len(dictionary1))  # 2 bars to consider
    width = WIDTH_BAR

    first_vals = []
    if dictionary2 is not None:
        second_vals = []

    if dictionary3 is not None:
        third_vals = []
        col1 = "#686158" #'#306998'
        col2 = "#BB8F55" # '#FFD43B'
        col3 = "#E18C18" #'#646464'
    else:
        col1 = "#E8E540"  #"#70615D" # "#28d9c2"
        col2 = "#39380F"  #"#E04625" #"#FF9933"
        col3 = None

    for key in dictionary1.keys():
        first_vals.append(dictionary1[key])
        if dictionary2 is not None:
            second_vals.append(dictionary2[key])
        if dictionary3 is not None:
            third_vals.append(dictionary3[key])

    rects1 = ax.bar(ind, first_vals, width, color=col1)
    if dictionary2 is not None:
        rects2 = ax.bar(ind + width, second_vals, width, color=col2)
        ax.legend((rects1[0], rects2[0]), (first_title, second_title), loc=loc)

    if dictionary3 is not None:
        rects3 = ax.bar(ind + 2 * width, third_vals, width, color=col3)
        ax.legend((rects1[0], rects2[0], rects3[0]), (first_title, second_title, third_title), loc=loc)

    if annotated:
        autolabel(rects1, ax)
        if dictionary2 is not None:
            autolabel(rects2, ax)
        if dictionary3 is not None:
            autolabel(rects3, ax)

    ax.set_ylabel(y_title)
    ax.set_xticks(range(len(dictionary1)))
    x_labels = list(dictionary1.keys())
    #x_labels = ["BasicNet"] + list(dictionary1.keys())[1:]
    ax.set_xticklabels(format_label_bar(x_labels))

    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


''' ---------------- format_label_bar -----------------------
This function reduces the size of the labels if it's too long
IN: current label
OUT: (new shorter) label
-------------------------------------------------------------'''


def format_label_bar(labels):
    labels_list = []
    for i, label in enumerate(labels):
        if 6 < len(str(label)):
            begin = INDEX_BEGIN_GRAPH_LABEL
            end = INDEX_END_GRAPH_LABEL
            labels_list.append(label[begin:end])
        else:
            labels_list.append(label)

    return labels_list


''' ---------------- autolabel ----------------------- '''


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 0.5 * height,
                str(round(height, 1)),
                ha='center', va='bottom')


''' ------------------------- store_in_csv -------------------------------------------
This function extends the csv file containing all the current 
results related to the different scenarios that were experimented 
IN: List of information to record about:
    data = [used_db, DIFF_FACES, WITH_PROFILE, DB_TRAIN]
    training = [pretrain_loss, nb_ep, bs, wd, lr, arch, opt, loss_type, margin]
    result = [losses_validation, f1_validation, f1_valid_detail, positive recall, f1_test]
    train_time: time required for training 
---------------------------------------------------------------------------------------'''


def store_in_csv(data, training, result, train_time):
    # Merge the loss and the weight info
    training[-2] = training[-2] + "_" + str(training[-1])
    training.pop()

    param = {}
    param["Non-pretrained Model"] = [("ds_" + data[0])] + data[1:] + ["not_pret"] + training[1:]
    param["Pretrained Model"] = [("ds_" + data[0])] + data[1:] + training

    lines = ["Non-pretrained Model", "Pretrained Model"] if 0 < len(result[0]["Non-pretrained Model"]) else [
        "Pretrained Model"]
    best_f1 = 0

    with open(CSV_NAME, 'a') as f:
        writer = csv.writer(f, delimiter=";")
        for i, line in enumerate(lines):
            curr_eval = [float(result[0][line][0]), float(result[0][line][int(round(len(result[0][line])) / 2)]),
                         float(result[0][line][-1]),
                         float(result[1][line][0]), float(result[1][line][int(round(len(result[1]) / 2))]),
                         float(result[1][line][-1])]

            best_f1 = float(max(result[1][line]))
            best_epoch = [str(result[1][line].index(best_f1))]
            best_f1 = [str(best_f1)]
            writer.writerow(param[line] + curr_eval + best_f1 + best_epoch + [result[2][line][0], result[2][line][1],
                                                                              result[3][line], result[4][line]] + [
                                str(train_time)] + [str(result[0][line])] + [str(result[1][line])])

    return best_f1


''' -------------------- store_in_csv ----------------------------
This function extends the csv file containing all the current 
results related to the different scenarios that were experimented 
IN: List of information to record about:
    - The data: [nb_repet, gallery_size, nb_probes, im_per_pers, db_test]
    - The algo: [model name, thresh, distance metric]
    - The result: 
        - Performance using the "yes/no => voting" strategy 
        - Performance using the distance-based ranking  
-------------------------------------------------------------------'''


def fr_in_csv(data, algo, result):
    # titles = ["nb_repet", "gallery_size", "nb_probes", "im_per_pers", "db_test", "model name", "thresh",
    # "distance metric", "nb_correct_vote", "nb_correct_dist"]

    with open(FR_CSV_NAME, 'a') as f:
        writer = csv.writer(f, delimiter=";")
        # writer.writerow(titles)
        writer.writerow(data + algo + result)

    print("The results from the FR task have been registered \n")


'''------------------ visualization_train -------------------------------------------
IN: epoch_list: list of specific epochs
    loss_list: list of lists of all the losses during each epoch
--------------------------------------------------------------------------------------'''


def visualization_train(num_epoch, losses_dic, save_name=None):
    key0 = list(losses_dic.keys())[0]  # pretrained
    key1 = list(losses_dic.keys())[1]  # non-pretrained

    # --------------------------------------------------------------------
    # Visualization of losses over iterations for different epochs
    # --------------------------------------------------------------------

    title = "Evolution of the loss for different epoches"
    try:
        epoch_list = range(0, num_epoch, math.ceil(num_epoch / 5))
        perc_train = [float(x) / len(losses_dic[key1][0]) for x in range(0, len(losses_dic[key1][0]))]
        dictionary = {}
        for i, epoch in enumerate(epoch_list):
            dictionary["epoch " + str(epoch)] = losses_dic[key1][epoch]
        multi_line_graph(dictionary, perc_train, title, x_label="percentage of data", y_label="Loss",
                         save_name=save_name + ".png", loc='upper right')
    except IndexError:
        print("The size of loss list is " + str(len(losses_dic[key1])))
        print("Epoch list is " + str(epoch_list))

    # --------------------------------------------------------------------
    # Visualization of losses over epochs
    # --------------------------------------------------------------------
    title_loss = "Evolution of the losses on training data"
    epoches = list(range(0, num_epoch, 1))

    # Compute the avg of each list in value
    for mode, list_losses in losses_dic.items():
        for i, losses_list in enumerate(losses_dic[mode]):
            losses_dic[mode][i] = np.mean(losses_list)

    # ------------- CASE 1: only pretrained scenario to expose --------------------
    if len(losses_dic[key1]) == 0:
        line_graph(range(0, len(losses_dic[key0]), 1), losses_dic[key0], "Loss according to the epochs",
                   x_label="Epoch",
                   y_label="Loss", save_name=save_name + "_loss.png")

    # ------------- CASE 2: both non-pretrained and pretrained scenario to expose --------------------
    else:
        multi_line_graph(losses_dic, epoches, title_loss, x_label="epoch", y_label="Loss",
                         save_name=save_name + "_loss.png", loc='upper right')


'''------------------------------------ visualization_validation -------------------------------------------- 
IN : self.losses_validation = {"Pretrained Model": [], "Non-pretrained Model": []}
     self.f1_validation = {"Pretrained Model": [], "Non-pretrained Model": [], "On Training Set": []}
------------------------------------------------------------------------------------------------------------ '''


def visualization_validation(loss, f1, acc, num_ep, save_name=None):
    title_loss = "Evolution of the losses"
    title_f1 = "Evolution of the f1-measure on the validation set"
    title_acc = "Evolution of the accuracy on the validation set"

    title_f1_train_valid = "Evolution of the f1-measure for the training and the validation set"

    key0 = list(loss.keys())[0]
    key1 = list(loss.keys())[1]
    key2 = list(f1.keys())[2]

    if len(loss[key1]) == 0:
        line_graph(range(0, len(loss[key0]), 1), loss[key0], "Loss according to the epochs", x_label="Epoch",
                   y_label="Loss", save_name=save_name + "_loss.png")
        line_graph(range(0, len(f1[key0]), 1), f1[key0], "f1-measure according to the epochs", x_label="Epoch",
                   y_label="f1-measure", save_name=save_name + "_f1.png")
        line_graph(range(0, len(acc[key0]), 1), acc[key0], "Accuracy according to the epochs", x_label="Epoch",
                   y_label="Accuracy", save_name=save_name + "_acc.png")
    else:
        dictionary_loss = {key0: loss[key0], key1: loss[key1]}
        dictionary_f1_valid = {key0: f1[key0], key1: f1[key1]}
        dictionary_acc_valid = {key0: acc[key0], key1: acc[key1]}
        dict_f1_valid_train = {key1: f1[key1], key2: f1[key2]}
        epoches = list(range(0, num_ep, 1))

        multi_line_graph(dictionary_loss, epoches, title_loss, x_label="epoch", y_label="Loss",
                         save_name=save_name + "_loss.png", loc='upper right')
        multi_line_graph(dictionary_f1_valid, epoches, title_f1, x_label="epoch", y_label="f1",
                         save_name=save_name + "_f1.png")
        multi_line_graph(dictionary_acc_valid, epoches, title_acc, x_label="epoch", y_label="acc",
                         save_name=save_name + "_acc.png")

        try:
            multi_line_graph(dict_f1_valid_train, epoches, title_f1_train_valid, x_label="epoch", y_label="f1",
                             save_name=save_name + "_f1_train_val.png")
        except ValueError:
            pass  # Case where there was no evaluation on the training set while training


if __name__ == '__main__':
    pass