import tools
from pathlib import Path
import datetime
import logging
import numpy as np
import sys
from init import setup_agent
from init import setup
import torch


class Learner:
    def __init__(self, args):
        path = Path(__file__).parent
        Path(path / 'logs').mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        now_but_text = "/logs/" + str(now.date()) + '-' + str(now.hour) + str(now.minute)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            filename=(str(path) + now_but_text + "-learner" + "-log.txt"),
                            filemode='w')
        logger = tools.get_writer()
        sys.stdout = logger

        partition_candidate = None
        terminating = False
        dmax = np.NINF
        distance = np.NINF
        agent = setup_agent(args[1])

        # TODO set partition memory

        i = 0
        while True:
            i += 1
            # TODO get replay batch
            agent.update(replay_batch)

            if i % 1000 == 0:
                pass
                # TODO update partition memory

            if i % 1000 == 0:
                agent.update_targets()