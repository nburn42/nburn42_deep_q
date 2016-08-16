"""
Deep Q Network
for open ai gym

Written By: Nathan Burnham, nburn42@gmail.com
"""
import gym

import tensorflow as tf
import numpy as np

import random

import neural_network
import gym_runner

        

def main():
    #game = "Pendulum-v0"
    #game = "MountainCar-v0"
    game = "CartPole-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/cartpole-experiment-1')

    layer_param_list = []

    layer_param_list.append(
        neural_network.RELULayerParams(100, name="hl1"))

    value_param_list = []
    
    value_param_list.append(
        neural_network.RELULayerParams(100, name="value1"))
    
    advantage_param_list = []

    advantage_param_list.append(
            neural_network.RELULayerParams(100, name="adv1"))
    
    duel_layer_params = neural_network.DuelLayersParams()
    duel_layer_params.value_layers = value_param_list
    duel_layer_params.advantage_layers = advantage_param_list
    
    layer_param_list.append(duel_layer_params)

    params = neural_network.Deep_Q_Params()
    params.env = env
    params.layer_param_list = layer_param_list

    nn = neural_network.Deep_Q(params)

    training_params = gym_runner.Training_Params()

    runner = gym_runner.Gym_Runner(env, nn)

    runner.train(training_params)
    #env.monitor.close()


if __name__ == "__main__":
    main()
