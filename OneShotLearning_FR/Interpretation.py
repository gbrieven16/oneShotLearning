import pandas as pd

#########################################
#       GLOBAL VARIABLES                #
#########################################


#########################################
#       FUNCTIONS                       #
#########################################


def find_highest(csv_name, crits):
    # ------------------------------
    # Open CSV
    # ------------------------------
    df = pd.read_csv(csv_name, delimiter=";")
    #print(df.keys())

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


""" ------------------------ find_optimal_val --------------------------------------------
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
    #print(df.keys())

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
