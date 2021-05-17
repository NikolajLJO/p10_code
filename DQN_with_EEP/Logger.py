import os
import csv
import pickle
from datetime import datetime


class Logger:
    def __init__(self):
        self.foldername = datetime.now().strftime("%d-%m-%Y_%H-%M")
        if not os.path.exists(os.path.join(self.foldername)):
            os.mkdir(os.path.join(self.foldername))

    # logsfiles
    def save_to_file(self, text_to_write, filename: str):
        with open(os.path.join(self.foldername, filename + ".csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow(text_to_write)

    # model
    def save_model(self, model, env_name: str):
        pickle.dump(model, open(os.path.join(self.foldername, env_name + "_DQN_agent_save.pickle"), "wb"))
        print("Saved " + env_name + "_DQN_agent_save.pickle succes")

    @staticmethod
    def load_model(path: str):
        if os.path.exists(path):
            model = pickle.load(open(path, "rb"))
            print(path)
            return model
        else:
            print(path + "does not exist")
