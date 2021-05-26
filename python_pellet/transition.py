from enum import IntEnum

'''
This class encabsulates an enumartion with values equal to the indicies of a transition.
'''

class Transition(IntEnum):
    STATE = 0
    ACTION = 1
    VISITED_BEFORE = 2
    AUXILIARY_REWARD = 3
    REWARD = 4
    IS_TERMINAL = 5
    STATE_PRIME = 6
    VISITED_AFTER = 7
    MC_REWARD = 8
