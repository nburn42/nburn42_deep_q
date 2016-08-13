import deep_q
import gym

def main():
    game = "Breakout-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/2lunar-lander-v2')


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

    nn = deep_q.DuelDualQ(env, layer_param_list)

    training_params = deep_q.Training_Params()
    training_params.max_episode = 100000000
    training_params.max_step = 10000
    training_params.show_freq = 1

    nn.train(training_params)
    #env.monitor.close()

if __name__ == "__main__":
    main()

