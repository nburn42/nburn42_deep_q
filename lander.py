import deep_q
import gym

def main():
    #game = "Acrobot-v0"
    #game = "MountainCar-v0"
    #game = "CartPole-v0"
    game = "LunarLander-v2"
    env = gym.make(game)
    #env.monitor.start('/tmp/lunar-lander-v2')
    nn = deep_q.NeuralNetwork(env)
    nn.train(show_display=True, 
            max_episode = 20000, 
            max_step = 999)
    #env.monitor.close()

if __name__ == "__main__":
    main()

