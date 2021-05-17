import csv
import numpy as np


def loss(pred, target):
    res = 0
    for i in range(len(pred)):
        p = np.longdouble(pred[i])
        t = np.longdouble(target[i])
        if abs(p - t) < 1:
            res += 0.5 * (p - t)**2
        else:
            res += abs(p - t) - 0.5
    return res / len(pred)


with open("DQN/backwardsLog.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    for i, row in enumerate(reader):
        result = np.longdouble(row[0])
        pred = row[1].replace("[", "").replace("]", "").split(",")
        target = row[2].replace("[", "").replace("]", "").split(",")

        lossval = loss(pred, target)
        if round(result, 5) != round(lossval, 5):
            print("row: |{0}| torch: |{1}| calculated: |{2}| difference: |{3}|".format(
                str.rjust(str(i), 6),
                str.rjust(str(result), 10),
                str.rjust(str(lossval), 10),
                str.rjust(str(abs(result - lossval)), 10)))
