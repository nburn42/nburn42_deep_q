import deep_q
import gym

def main():
    game = "Breakout-v0"
    env = gym.make(game)
    env.monitor.start('/tmp/breakout')


    layer_param_list = []
    
    layer_param_list.append(
            deep_q.ConvolutionalLayerParams(patch_size=8, stride=4, features=32, pool_size=2,  name="conv1"))
    
    layer_param_list.append(
            deep_q.ConvolutionalLayerParams(patch_size=4,  stride=2, features=64, pool_size=2, name="conv2"))

    layer_param_list.append(
            deep_q.ConvolutionalLayerParams(patch_size=3, stride=1, features=64, pool_size=2, name="conv3"))
    
    layer_param_list.append(
            deep_q.RELULayerParams(neurons=100, name="relu1"))

    layer_param_list.append(
            deep_q.RELULayerParams(neurons=100, name="relu2"))

    value_param_list = []

    value_param_list.append(
        deep_q.RELULayerParams(100, name="value1"))

    value_param_list.append(
        deep_q.RELULayerParams(100, name="value2"))

    advantage_param_list = []

    advantage_param_list.append(
            deep_q.RELULayerParams(100, name="adv1"))

    advantage_param_list.append(
            deep_q.RELULayerParams(100, name="adv2"))

    duel_layer_params = deep_q.DuelLayersParams()
    duel_layer_params.value_layers = value_param_list
    duel_layer_params.advantage_layers = advantage_param_list

    layer_param_list.append(duel_layer_params)
    
    nn = deep_q.DuelDualQ(env, layer_param_list)

    training_params = deep_q.Training_Params()
    training_params.max_episode = 100000000
    training_params.max_step = 999
    training_params.show_freq = 5
    training_params.memory_size = 20000
    training_params.train_freq = 64
    training_params.batch_size = 150
    training_params.random_decay = .9995
    training_params.update_param_freq = 10000

    nn.train(training_params)
    env.monitor.close()

if __name__ == "__main__":
    main()

