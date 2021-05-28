from typing import Mapping
from matplotlib import pyplot as plt
from tools import process_score_over_steps
import csv
from matplotlib.ticker import FuncFormatter




def millions_formatter(x, pos):
    return f'{x / 1000000}'


if __name__ == "__main__":

    process_scores = input("Do you want to process scores over steps (y/n)? ")
    if process_scores == "y":
        filename = input("Enter the name of the txt file WITHOUT .txt")
        process_score_over_steps(filename)
    
    else:
        csv_filename = input("Enter the file name of the csv file WITH .csv: ")
        x_values = []
        y_values = []
        with open("logs/" + csv_filename, newline='') as file_handle:
            for row in csv.reader(file_handle):
                x_values.append(int(row[0]))
                y_values.append(int(row[1]))
        
        #x_values = []
        #for i in range(1,len(y_values)+1):
        #   x_values.append(i)

        x_label = input("Enter the label for the x-axis of the graph: ")
        y_label = input("Enter the label for the y-axis of the graph: ")
        graph_title = input("Enter the title of the graph: ")
        fig, ax1 = plt.subplots()
        ax1.plot(x_values, y_values)
        ax1.set_xlabel(x_label, fontsize="14")
        ax1.set_ylabel(y_label, fontsize="14")
        ax1.set_title(graph_title, fontsize="14")
        ax1.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        plt.show()
    