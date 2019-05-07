import pandas as pd
from Visualization import bar_chart, line_graph
import numpy as np

#########################################
#       GLOBAL VARIABLES                #
#########################################


#########################################
#       FUNCTIONS                       #
#########################################

""" ------------------------ to_dic -------------------------------------
This function build a dictionary from the content of the csv_name
where the keys are the different values for crit_key and the value 
are the mean over the corresponding values of crit_value

Eg: crit_key ; crit_value
    1        ; 3
    1        ; 7
    3        ; 3
    1        ; 2
    => {1: 4, 3: 3}

IN: lower_bound: the minimum value the value of crit_value has to have 
to be considered 
crit key: (list of 2) crit to discuss (the first one represent the bars inside 
a set and the second element represent the different sets of bars) 
------------------------------------------------------------------------ """


def to_dic(csv_name, crit_key_list, crit_value, lower_bound=0):
    # ------------------------------
    # Open CSV
    # ------------------------------
    df = pd.read_csv(csv_name, delimiter=";")

    dic = {}

    # ------------------------------------------------------------------------
    # Store all values of crit_value to the corresponding crit_key
    # ------------------------------------------------------------------------
    for i, row in df.iterrows():
        if lower_bound < row[crit_value]:
            try:
                if type(crit_key_list) is list:
                    dic[row[crit_key_list[0]]][row[crit_key_list[1]]].append(row[crit_value])
                else:
                    dic[row[crit_key_list]].append(row[crit_value])

            except KeyError:
                if type(crit_key_list) is list:
                    try:
                        dic[row[crit_key_list[0]]][row[crit_key_list[1]]] = [row[crit_value]]
                    except KeyError:
                        dic[row[crit_key_list[0]]] = {}
                        dic[row[crit_key_list[0]]][row[crit_key_list[1]]] = [row[crit_value]]
                else:
                    dic[row[crit_key_list]] = [row[crit_value]]

    # ---------------------------------------------------------------------------
    # Compute the avg of all values of crit_value corresponding to each crit_key
    # ---------------------------------------------------------------------------
    for key_val, val_list in dic.items():
        if type(val_list) is dict:
            for key2, values in val_list.items():
                dic[key_val][key2] = sum(values) / float(len(values))
        else:
            dic[key_val] = sum(val_list) / float(len(val_list))

    return dic

"""
This function returns a dictionary (of dictionaries) built from dic 
where the keys are the elements of the list keys1 and the values are 
the values in dic 
"""
def key_restiction(dic, keys1, keys2=None):
    filtered_dic = {}
    for key, value in dic.items():
        if key in keys1:
            if keys2 is not None:
                for key2, value2 in value.items():
                    if key2 in keys2:
                        try:
                            filtered_dic[key][key2] = value2
                        except KeyError:
                            filtered_dic[key] = {}
                            filtered_dic[key][key2] = value2


            else:
                filtered_dic[key] = value
    return filtered_dic


""" ------------------------ find_highest ----------------------------------------
This function returns the criterion in crits having the highest sum of values over
all the rows 
------------------------------------------------------------------------------------ """


def find_highest(csv_name, crits):
    # ------------------------------
    # Open CSV
    # ------------------------------
    df = pd.read_csv(csv_name, delimiter=";")
    # print(df.keys())

    # ------------------------------
    # Compute a score for each line
    # ------------------------------
    scores = {}  # list where the key is the crit and the value is the corresponding total
    for i, row in df.iterrows():
        for j, crit in enumerate(crits):
            try:
                scores[crit] += row[crit]
            except KeyError:
                scores[crit] = row[crit]

    return max(scores, key=scores.get)


""" ------------------------ find_optimal_val ----------------------------------------
This function returns the value of to_return maximizing the values of the criteria
in eval_crit 

IN: csv_name: name of the csv file to consider
    to return: name of the feature whose optimal value has to be returned
    eval_crit: list of feature whose value evaluate the current scenario 
------------------------------------------------------------------------------------- """


def find_optimal_val(csv_name, to_return, eval_crit):
    # ------------------------------
    # Open CSV
    # ------------------------------
    df = pd.read_csv(csv_name, delimiter=";")
    # print(df.keys())

    # ------------------------------
    # Compute a score for each line
    # ------------------------------
    scores = []  # list where the index is the associated to a row and the value is the corresponding score
    for i, row in df.iterrows():
        score = 0

        for i, crit in enumerate(eval_crit):
            score += float(row[crit])
        scores.append(score)

    # -------------------------------------------
    # Compute a score for each value of to_return
    # -------------------------------------------
    possible_values = list(df[to_return].unique())
    print(possible_values)
    score_per_val = {}
    for j, value in enumerate(possible_values):
        df_val = df[df[to_return] == value]
        score_per_val[value] = 0
        for i, row in df_val.iterrows():
            score_per_val[value] += scores[i]
        try:
            score_per_val[value] /= len(df_val)
        except ZeroDivisionError:
            print("ERR: No row matched with value " + str(value))

    # Extract the key having the highest value
    return max(score_per_val, key=score_per_val.get)


"""
IN: restriction_min: dictionary where the key is the criterion the restriction is on and the value 
is the minimum value this criterion has to have 
restriction_equal: dictionary where the key is the criterion the restriction is on and the value 
is a list of values the criterion has to be equal to 
"""


def print_with_best_scenarios(csv_name, restriction_min, restriction_equal, to_visualize=None):
    # ------------------------------
    # Open CSV
    # ------------------------------
    df = pd.read_csv(csv_name, delimiter=";")
    best_rows = []

    # ------------------------------
    # Restriction Check
    # ------------------------------
    for i, row in df.iterrows():
        to_add = True
        # ----------- Restriction MIN Check -----------
        for crit, min_val in restriction_min.items():
            if row[crit] < min_val or is_not_nb(row[crit]): # np.isnan(row[crit]):
                to_add = False
                break

        # ----------- Restriction Equal Check -----------
        for crit, list_val in restriction_equal.items():
            if row[crit] not in list_val:
                to_add = False
                break

        if to_add:
            best_rows.append(row)

    # ------------------------------
    # Visualize row
    # ------------------------------
    for i, row in enumerate(best_rows):
        if to_visualize is None:
            print("\n" + str(row))
        else:
            print("\n---------- For row " + str(i) + ": ----------")
            for j, crit in enumerate(to_visualize):
                print("Value of " + crit + " is " + str(row[crit]))


def is_not_nb(nb):
    return not (float('-inf') < float(nb) < float('inf'))

# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    csv_name = "result/fr_model_evaluation.csv"
    test_id = 3

    if test_id == 1:
        print(" -------------------------- TEST 1 --------------------------------")
        print("Visualize the best value for to_return according to the eval crit")
        print(" ------------------------------------------------------------------")
        eval_crit = ["nb_correct_vote", "nb_correct_dist", "EER"]
        to_return = "thresh"  # "distance metric" #"im_per_pers"   # "thresh"
        print("The optimal value is " + str(find_optimal_val(csv_name, to_return, eval_crit)) + " for " + to_return)

    if test_id == 2:
        print(" -------------------------- TEST 2 --------------------------------")
        print("Visualize the criterion having the highest value on average")
        print(" ------------------------------------------------------------------")
        crits = ["nb_correct_vote", "nb_correct_dist"]
        print("The optimal crit is " + str(find_highest(csv_name, crits)))

    if test_id == 3:
        print(" -------------------------- TEST 3 --------------------------------")
        print("Visualize the f1 measure for different architectures and losses")
        print(" ------------------------------------------------------------------")
        csv_name = "result/model_evaluation_test.csv"
        title = "Comparison between different architectures and losses"
        # Compare architectures
        di1 = to_dic(csv_name, ["LossType - Weighted Classes", "Archit"], "best_f1_score", lower_bound=0)
        filt_di1 = key_restiction(di1, ["triplet_loss", "cross_entropy", "constrastive_loss"],
                                  keys2=["1default", "4AlexNet", "VGG16"])
        print(filt_di1)
        # arch = list(di1.keys())
        bar_chart(filt_di1["triplet_loss"], filt_di1["cross_entropy"], title, dictionary3=filt_di1["constrastive_loss"],
                  first_title="triplet_loss", second_title="cross_entropy", third_title="constrastive_loss",
                  annotated=True, y_title="f1 measure", save_name="arch_loss_comp")

    if test_id == 4:
        print(" -------------------------- TEST 4 --------------------------------")
        print("Visualize the \"best\" scenarios")
        print(" ------------------------------------------------------------------")
        csv_name = "model_evaluation.csv"
        restriction_min = {"Nb of triplets (train)": 1000, "NbEpoches": 40, "best_f1_score": 88} #, "f1_test": 60}
        restriction_equal = {"Optimizer": ["Adam"]}
        to_visualize = ["Archit", "LossType - Weighted Classes", "best_f1_score", "With Pretraining", "Name BD",
                        "Nb of triplets (train)", "NbEpoches"]
        print_with_best_scenarios(csv_name, restriction_min, restriction_equal, to_visualize=to_visualize)

    if test_id == 5:
        # -------------------------------------------------------
        # Compare the scenarios with and without synthetic data
        # -------------------------------------------------------
        csv_name = "result/syntAndReal_old.csv"
        title = "Comparison between different data quantities and nature"
        # Compare architectures
        di1 = to_dic(csv_name, ["Synth", "Total Nb of Images"], "f1_score3", lower_bound=0)

        print(di1)
        # arch = list(di1.keys())
        bar_chart(di1["real"], di1["real + synth"], title, first_title="real", second_title="real + synth",
                  annotated=False, y_title="f1 measure", save_name="synthVSreal")

    if test_id == 6:
        print(" -------------------------- TEST 6 --------------------------------")
        print("Visualize the best distance Metric")
        print(" ------------------------------------------------------------------")
        restriction_equal = {"gallery_size": [50], "nb_probes":[20], "thresh": [5]}
        title = "Accuracy according to the distance metric"
        di1 = to_dic(csv_name, ["with z=fr", "distance metric"], "nb_correct_dist", lower_bound=0)
        print(di1)
        # arch = list(di1.keys())
        bar_chart(di1["True"], di1["False"], title, first_title="Triplet FR based", second_title="Generator FR based",
                  annotated=False, y_title="Accuracy", save_name="fr_dist")

    if test_id == 7:
        print(" -------------------------- TEST 7 --------------------------------")
        print("Visualize the performance according to the gallery size")
        print(" ------------------------------------------------------------------")
        x = [20, 40, 60, 80, 100, 120, 150, 200, 300, 400]
        di = to_dic(csv_name, ["gallery_size"], "nb_correct_dist", lower_bound=0)
        y = []
        for i, gal_size in enumerate(x):
            y.append(di[gal_size])
        title = "Accuracy according to the gallery size"

        line_graph(x, y, title, x_label="x", y_label="y", save_name=None)