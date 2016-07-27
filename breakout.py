import deep_q
import gym

def main():
    game = "Breakout-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/breakout-v0')
    training_params = deep_q.NeuralNetwork.training_params
    training_params['TRAIN_FREQ'] = 50
    training_params['MEMORY_SIZE'] = 20000
    training_params['BATCH_SIZE'] = 100
    training_params['DISPLAY_FREQ'] = 1
    training_params['RANDOM_DECAY'] = .9999
    # need min random so the network will
    # hit the fire key to launch the ball
    training_params['RANDOM_MIN'] = .05
    nn = deep_q.NeuralNetwork(env, training_params=training_params)

    nn.run(show_display=True, 
            max_episode = 2000000, 
            max_step = 999)
    #env.monitor.close()

if __name__ == "__main__":
    main()

