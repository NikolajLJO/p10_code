from Qlearning import QLearning
import DQN_testing
from setup import Setup


if __name__ == "__main__":
    parser = Setup()
    args = parser.run()
    env_names = ['BreakoutDeterministic-v0', 'SpaceInvadersDeterministic-v4', 'MsPacmanDeterministic-v4',
                 'MontezumaRevengeDeterministic-v4', 'EnduroDeterministic-v4', 'BattleZoneDeterministic-v4',
                 'RoadRunnerDeterministic-v4', 'VideoPinballDeterministic-v4']
    if args.test:
        load_path = input("write path to folder containing the models:")
        DQN_testing.test(load_path)
    else:
        for env_name in env_names:
            # TODO make these inputs come from terminal as options and have the values be the default constructor options
            agent = QLearning(env_name, episode_steps=4500, stack_count=4, explore_steps=2250000, double=True,
                              target_update_frequency=1000, max_memory=1000000, update_frequency=4, max_steps=10000000,
                              learning_rate=0.0002, initial_eps=1.0, trained_eps=0.01, final_eps=0.01, eps_initial_frames=1000000,
                              print_freq=20)
            if args.target_fq:
                agent.target_update_frequency = args.target_fq
            if args.explore:
                agent.explore_steps = args.explore
            if args.episode_steps:
                agent.episode_steps = args.episode_steps
            if args.episode_num:
                agent.episode_num = args.episode_num
            if args.env:
                pass  # TODO IMPL CHNAGE ENV HERE, ENV is ised in DQNtesting.test
            agent.run()
