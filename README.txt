Source code for the AAAI 2019 paper 'Deriving Subgoals Autonomously to Accelerate Learning in Sparse Reward Domains' by Michael Dann, Fabio Zambetta and John Thangarajah.


CREDITS:
Our code builds off DeepMind's original DQN agent (https://sites.google.com/a/deepmind.com/dqn/) but has been adapted to work with v0.6 of the Arcade Learning Environment (https://github.com/mgbellemare/Arcade-Learning-Environment/releases).


WARNING:
The code is still very much in 'academic' form... There are plenty of junk/unused variables, old comments that are no longer applicable, variable names that don't tie in properly with the terminology of the paper, etc. In it's current form, this repository is not suited to much besides reproducing the results in our paper. We intend to release a cleaner and more extensible version soon! In the meantime, we encourage researchers who are interested in collaborating and/or extending our work to contact Michael (Michael.Dann@rmit.edu.au).


RUNNING THE AGENT:
1. Follow the instructions in ALE-README.txt to install the Arcade Learning Environment v0.6. Make sure you install the python module too!
2. cd src/agent
3. ./run_gpu_pellet montezuma_revenge (this will run the pellet rewards agent on Montezuma's Revenge. To run the agent without pellets, use the script 'run_gpu_baseline'. To run a different game, choose one of the other titles in the 'roms' directory.


TIPS:
- If you run into memory issues, you might need to reduce the size of the replay buffer. This can be done by editing the 'replay_memory' parameter in the 'run_gpu_pellet' (or 'run_gpu_baseline').
