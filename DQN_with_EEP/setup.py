import argparse


class Setup:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def run(self):
        self.parser.add_argument("--test", help="flag indicating if i should load values fomr file",
                                 action="store_true")
        self.parser.add_argument("--explore", help="amount of explore steps before learning",
                                 type=int)
        self.parser.add_argument("--target_fq", help="frequency of updating the target network in steps",
                                 type=int)
        self.parser.add_argument("--episode_num", help="number of episodes to train on",
                                 type=int)
        self.parser.add_argument("--episode_steps", help="number of maximum steps for an episode",
                                 type=int)
        self.parser.add_argument("--env", help="enviroment name")
        return self.parser.parse_args()
