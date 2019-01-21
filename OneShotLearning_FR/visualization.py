import matplotlib.pyplot as plt
import pickle
import numpy as np

#  ===============================================================
#                       GLOBAL VARIABLES
# ================================================================


NB_DATA_GRAPH = 1000
MAX_NB_KEYS = 12

INDEX_BEGIN_GRAPH_LABEL = 0
INDEX_END_GRAPH_LABEL = 9


# ================================================================
#         IMPLEMENTATION OF THE VISUALIZATION TOOLS
# ================================================================


def line_graph(x, y, title, x_label="x", y_label="y"):
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def multi_line_graph(dictionary, x_elements, title, x_label="x", y_label="Score"):
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

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # --------- Set the fontsize ------------------
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # --------- Save -----------
    plt.show()
    plt.close()


# -------------- bar_chart --------------------
# IN: dictionary1 and dictionary2 have the same
#     keys and 1 value in a list
# ---------------------------------------------

def bar_chart(dictionary1, dictionary2, title, dictionary3=None, first_title='Average', second_title='Std',
              annotated=True, y_title="Response time (ms)"):
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



if __name__ == '__main__':

    # Retrieve information about the training phase
    with open("losses_train_16.txt", "rb") as fp:  # UnPickling
        losses_train = pickle.load(fp)
    with open("losses_test_16.txt", "rb") as fp:  # UnPickling
        losses_test = pickle.load(fp)
    with open("acc_test_16.txt", "rb") as fp:  # UnPickling
        acc_test = pickle.load(fp)