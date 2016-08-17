import neural_network
import gym_runner
import gym

def main():
    #game = "Acrobot-v0"
    #game = "MountainCar-v0"
    #game = "CartPole-v0"
    game = "LunarLander-v2"
    env = gym.make(game)
    env.monitor.start('/tmp/lunar-lander-v2')

    layer_param_list = []
    
    layer_param_list.append(
            neural_network.RELULayerParams(neurons=100, name="relu1"))

    layer_param_list.append(
            neural_network.RELULayerParams(neurons=100, name="relu2"))

    layer_param_list.append(
            neural_network.RELULayerParams(neurons=100, name="relu3"))

    value_param_list = []
    
    value_param_list.append(
        neural_network.RELULayerParams(100, name="value1"))
    
    value_param_list.append(
        neural_network.RELULayerParams(100, name="value2"))

    advantage_param_list = []

    advantage_param_list.append(
            neural_network.RELULayerParams(100, name="adv1"))
    
    advantage_param_list.append(
            neural_network.RELULayerParams(100, name="adv2"))

    duel_layer_params = neural_network.DuelLayersParams()
    duel_layer_params.value_layers = value_param_list
    duel_layer_params.advantage_layers = advantage_param_list
    
    layer_param_list.append(duel_layer_params)
    
    params = neural_network.Deep_Q_Params()
    params.env = env
    params.layer_param_list = layer_param_list
    params.train_freq = 64
    params.batch_size = 2000
    params.update_param_freq = 1000

    nn = neural_network.Deep_Q(params)

    params = gym_runner.Training_Params()
    params.max_episode = 100000000
    params.max_step = 999
    params.show_freq = 10
    params.memory_size = 200000
    params.random_decay = .999

    gym_runner.Gym_Runner(env, nn).train(params)
    env.monitor.close()

if __name__ == "__main__":
    main()

