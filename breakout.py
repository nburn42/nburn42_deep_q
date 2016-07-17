import deep_q
import gym

def main():
    game = "Breakout-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/breakout-v0')
    training_params = deep_q.NeuralNetwork.training_params
    training_params["MEMORY_SIZE"] = 2000
    training_params["BATCH_SIZE"] = 50
    training_params["DISPLAY_FREQ"] = 1
    nn = deep_q.NeuralNetwork(env, training_params=training_params)

    nn.train(show_display=True, 
            max_episode = 20000, 
            max_step = 999)
    #env.monitor.close()

if __name__ == "__main__":
    main()

