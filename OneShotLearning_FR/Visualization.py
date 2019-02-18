import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import csv
import numpy as np

# ================================================================
#                       GLOBAL VARIABLES
# ================================================================

CSV_NAME = "model_evaluation.csv"
NB_DATA_GRAPH = 1000
MAX_NB_KEYS = 12

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


def multi_line_graph(dictionary, x_elements, title, x_label="x", y_label="Score", save_name=None):
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # --- We go through all the line to represent ------
    for line in dictionary.keys():
        # ------ Plot the evolution of the value over x elements --------
        plt.plot(x_elements, dictionary[line], label=str(line))

    # --------- Legend -----------------------------

    legend = plt.legend(loc='lower right', shadow=True)
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
              annotated=True, y_title="Response time (ms)", save_name=True):
    if len(dictionary1) == 0:
        print("INFO: In Visualization: Empty dictionaries")
        return None

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(dictionary1))  # 2 bars to consider
    width = 0.4

    first_vals = []
    if dictionary2 is not None:
        second_vals = []

    if dictionary3 is not None:
        third_vals = []

    for key in dictionary1.keys():
        first_vals.append(dictionary1[key])
        if dictionary2 is not None:
            second_vals.append(dictionary2[key])
        if dictionary3 is not None:
            third_vals.append(dictionary3[key])

    rects1 = ax.bar(ind, first_vals, width, color='#FF8C00')
    if dictionary2 is not None:
        rects2 = ax.bar(ind + width, second_vals, width, color='#255D79')
        ax.legend((rects1[0], rects2[0]), (first_title, second_title))

    if dictionary3 is not None:
        rects3 = ax.bar(ind + 2 * width, third_vals, width, color='#255D79')
        ax.legend((rects1[0], rects2[0], rects3[0]), (first_title, second_title, "Data Quantity"))

    if annotated:
        autolabel(rects1, ax)
        if dictionary2 is not None:
            autolabel(rects2, ax)
        if dictionary3 is not None:
            autolabel(rects3, ax)

    ax.set_ylabel(y_title)
    ax.set_xticks(range(len(dictionary1)))
    ax.set_xticklabels(format_label_bar(dictionary1.keys()))

    plt.title(title)
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


''' -------------------- store_in_csv ----------------------------
This function extends the csv file containing all the current 
results related to the different scenarios that were experimented 
IN: List of information to record about:
    data = [used_db, DIFF_FACES, WITH_PROFILE, DB_TRAIN]
    training = [pretrain_loss, nb_ep, bs, wd, lr, arch, opt, loss_type, margin]
    result = [losses_test, f1_test]
-------------------------------------------------------------------'''


def store_in_csv(data, training, result):

    if training[-2] == "triplet_loss":
        training[-2] = training[-2] + "_" + str(training[-1])
    training.pop() # Remove margin information from the list
    curr_parameters = [("ds_" + data[0])] + data[1:] + training

    curr_evaluation = [float(result[0][0]), float(result[0][int(round(len(result[0])) / 2)]), float(result[0][-1]),
                       float(result[1][0]), float(result[1][int(round(len(result[1]) / 2))]), float(result[1][-1])]

    best_f1 = float(max(result[1]))
    best_epoch = result[1].index(best_f1)
    print("IN STORE IN CSV: best f1 is " + str(best_f1))

    # titles = ["Name BD", "IsDiffFaces", "IsWithProfile", "Db_train", With Pretraining,
    #  "NbEpoches", "BS", "WD", "LR", "ArchType", "Optimizer", "LossType", "Weighted Classes",
    # "Loss1", "Loss2", "Loss3", "F11", "F12", 'F13', "F1Best", epochBest]

    with open(CSV_NAME, 'a') as f:
        writer = csv.writer(f, delimiter=";")
        # writer.writerow(titles)
        writer.writerow(curr_parameters + curr_evaluation + best_f1 + best_epoch)


'''------------------ visualization_train -------------------------------------------
IN: epoch_list: list of specific epochs
    loss_list: list of lists of all the losses during each epoch
--------------------------------------------------------------------------------------'''


def visualization_train(epoch_list, loss_list, save_name=None):

    title = "Evolution of the loss for different epoches"
    perc_train = [float(x) / len(loss_list[0]) for x in range(0, len(loss_list[0]))]
    dictionary = {}
    for i, epoch in enumerate(epoch_list):
        dictionary["epoch " + str(epoch)] = loss_list[epoch]

    multi_line_graph(dictionary, perc_train, title, x_label="percentage of data", y_label="Loss", save_name=save_name)


'''------------------ visualization_test ----------------------------- '''


def visualization_test(loss, f1, save_name=None):
    title_loss = "Comparison of the evolution of the losses"
    title_f1 = "Comparison of the evolution of the f1-measure"

    key1 = list(loss.keys())[0]
    key2 = list(loss.keys())[1]

    if len(loss[key2]) == 0:
        line_graph(range(0, len(loss[key1]), 1), loss[key1], "Loss according to the epochs", x_label="Epoch",
                   y_label="Loss", save_name=save_name + "_loss.png")
        line_graph(range(0, len(f1[key1]), 1), f1[key1], "f1-measure according to the epochs", x_label="Epoch",
                   y_label="f1-measure", save_name=save_name + "_f1.png")
    else:
        dictionary_loss = {key1: loss[key1], key2: loss[key2]}
        dictionary_f1 = {key1: f1[key1], key2: f1[key2]}
        epoches = list(range(0, len(loss[key1]), 1))

        multi_line_graph(dictionary_loss, epoches, title_loss, x_label="epoch", y_label="Loss",
                         save_name=save_name + "_loss.png")
        multi_line_graph(dictionary_f1, epoches, title_f1, x_label="epoch", y_label="f1",
                         save_name=save_name + "_f1.png")


if __name__ == '__main__':
    pass
