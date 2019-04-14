import pandas as pd

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
------------------------------------------------------------------------ """


def to_dic(csv_name, crit_key, crit_value, lower_bound=0):
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
                dic[crit_key].append(row[crit_value])
            except KeyError:
                dic[crit_key] = [row[crit_value]]

    # ---------------------------------------------------------------------------
    # Compute the avg of all values of crit_value corresponding to each crit_key
    # ---------------------------------------------------------------------------
    for key_val, val_list in dic.items():
        dic[key_val] = sum(val_list)/float(len(val_list))

    return dic


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


# ================================================================
#                    MAIN
# ================================================================


if __name__ == "__main__":

    csv_name = "test.csv"
    test_id = 2

    if test_id == 1:
        eval_crit = ["nb_correct_vote", "nb_correct_dist", "EER"]
        to_return = "thresh"  # "distance metric" #"im_per_pers"   # "thresh"
        print("The optimal value is " + str(find_optimal_val(csv_name, to_return, eval_crit)) + " for " + to_return)

    if test_id == 2:
        crits = ["nb_correct_vote", "nb_correct_dist"]
        print("The optimal crit is " + str(find_highest(csv_name, crits)))
