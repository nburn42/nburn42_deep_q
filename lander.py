import deep_q
import gym

def main():
    #game = "Acrobot-v0"
    #game = "MountainCar-v0"
    #game = "CartPole-v0"
    game = "LunarLander-v2"
    env = gym.make(game)
    #env.monitor.start('/tmp/lunar-lander-v2')
    training_params = deep_q.NeuralNetwork.training_params
    training_params['RANDOM_DECAY'] = .9999
    nn = deep_q.NeuralNetwork(env, training_params = training_params)
    nn.run(show_display=True, 
            max_episode = 20000, 
            max_step = 999)
    #env.monitor.close()

if __name__ == "__main__":
    main()

