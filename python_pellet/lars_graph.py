from typing import Mapping
from matplotlib import pyplot as plt
from tools import process_score_over_steps
import csv
from matplotlib.ticker import FuncFormatter




EPISODE_SCORE_INDEX = 1
AVERAGE_100_SCORE_INDEX = 5



def millions_formatter(x, pos):
    return f'{x / 1000000}'



def plot_graph(index, ax1):
    csv_filename = input("Enter the file name of the {}st csv file WITH .csv: ".format("index"))
    x_values = []
    y_values = []
    use_average_score = input("Do you want to create the graph over the averege score for 100 steps (y/n): ")

    if use_average_score == "y":
        with open("logs/" + csv_filename, newline='') as file_handle:
            for row in csv.reader(file_handle):
                x_values.append(int(row[0]))
                y_values.append(float(row[AVERAGE_100_SCORE_INDEX]))
    else:
        with open("logs/" + csv_filename, newline='') as file_handle:
            for row in csv.reader(file_handle):
                x_values.append(int(row[0]))
                y_values.append(int(row[EPISODE_SCORE_INDEX]))
    
    # To use episodes instead of steps uncomment these lines and remove above x_values.append(int(row[0]))
    #x_values = []
    #for i in range(1,len(y_values)+1):
    #   x_values.append(i)

    ax1.plot(x_values, y_values)
    


if __name__ == "__main__":
    
    process_scores = input("Do you want to process scores over steps (y/n)? ")
    if process_scores == "y":
        filename = input("Enter the name of the txt file: ")
        process_score_over_steps(filename)
    
    else:
        fig, ax1 = plt.subplots()
        amount = input("Enter the amount of lines you want to plot: ")
        for i in range(1,int(amount)+1):
            plot_graph(i, ax1)

        x_label = input("Enter the label for the x-axis of the graph: ")
        y_label = input("Enter the label for the y-axis of the graph: ")
        graph_title = input("Enter the title of the graph: ")
        ax1.set_xlabel(x_label, fontsize="14")
        ax1.set_ylabel(y_label, fontsize="14")
        ax1.set_title(graph_title, fontsize="14")
        ax1.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        plt.show()