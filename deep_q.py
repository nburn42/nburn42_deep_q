"""
Deep Q Network
for open ai gym

Written By: Nathan Burnham, nburn42@gmail.com
"""
import gym

import tensorflow as tf
import numpy as np

import random


class Layer:

    layer_description = ""
    
    training_weight = None
    training_bias = None

    target_weight = None
    target_bias = None

    def make_weights(self, size):
        return tf.Variable(tf.truncated_normal(
            shape=size,
            stddev=0.1))

    def make_bias(self, size, initial_bias):
        return tf.Variable(tf.constant(initial_bias,
            shape=size))
    
    def get_output_layer(self):
        return [self.training_output_layer, self.target_output_layer]

    def get_training_output_layer(self):
        return self.training_output_layer

    def get_target_output_layer(self):
        return self.target_output_layer

    def update_target(self, sess):
        self.target_weight.assign(self.training_weight.eval(session=sess)).eval(session=sess)
        self.target_bias.assign(self.training_bias.eval(session=sess)).eval(session=sess)
    
    def get_description(self):
        return self.layer_description


class ConvolutionalLayerParams:
    def __init__(self, patch_size, stride, features, pool_size, name="conv_layer"):
        self.patch_size = patch_size
        self.stride = stride
        self.features = features
        self.pool_size = pool_size
        self.name = name
        self.initial_bias = 1e-6

# currently will only do conv layers with 2x2 pooling layers after
# also will only take inputs with color broken out in a dimension
class ConvolutionalLayer(Layer):
    def make_conv(self, previous_layer, weight, stride):
        return tf.nn.conv2d(previous_layer, weight, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(self, previous_layer, patch_size = 2):
        return tf.nn.max_pool(previous_layer, ksize=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1], padding='SAME')
        
    def __init__(self, prev_layer, layer_params):
        self.training_input_layer, self.target_input_layer =prev_layer
        with tf.name_scope(layer_params.name):
            #conv_depth is the channel count of the image
            conv_depth = int(self.training_input_layer.get_shape()[3])
            weight_size = [layer_params.patch_size, layer_params.patch_size, conv_depth, layer_params.features] 
            bias_size = [layer_params.features]
            self.training_weight = self.make_weights(weight_size)
            self.target_weight = self.make_weights(weight_size)
            self.training_bias = self.make_bias(bias_size, layer_params.initial_bias)
            self.target_bias = self.make_bias(bias_size, layer_params.initial_bias)
            training_conv = tf.nn.relu(self.make_conv(self.training_input_layer, self.training_weight, layer_params.stride) + self.training_bias)
            target_conv = tf.nn.relu(self.make_conv(self.target_input_layer, self.target_weight, layer_params.stride) + self.target_bias)
            self.training_output_layer = self.max_pool(training_conv)
            self.target_output_layer = self.max_pool(target_conv)
            self.layer_description += "conv {} + {}    {}\npool {}\n".format(weight_size, bias_size, layer_params.name, layer_params.pool_size)
            
class RELULayerParams:
    def __init__(self, neurons, name="relu_layer", skip_relu = False):
        self.neurons = neurons
        self.name = name
        self.skip_relu = skip_relu
        self.initial_bias = 1e-6

class RELULayer(Layer):

    def __init__(self, prev_layer, layer_params):
        self.training_input_layer, self.target_input_layer = prev_layer
        with tf.name_scope(layer_params.name):
            #flatten network if not yet flattened
            if len(self.training_input_layer.get_shape()) > 2:
                flat_size = int(reduce(lambda x,y : x*y, self.training_input_layer.get_shape()[1:]))
                self.training_input_layer = tf.reshape(self.training_input_layer, [-1, flat_size])
                self.target_input_layer = tf.reshape(self.target_input_layer, [-1, flat_size])

            previous_size = int(self.training_input_layer.get_shape()[1])
            weight_size = [layer_params.neurons, previous_size]
            bias_size = [layer_params.neurons]
            self.training_weight = self.make_weights(weight_size)
            self.target_weight = self.make_weights(weight_size)
            self.training_bias = self.make_bias(bias_size, layer_params.initial_bias)
            self.target_bias = self.make_bias(bias_size, layer_params.initial_bias)

            self.training_output_layer = tf.matmul(self.training_input_layer, tf.transpose(self.training_weight)) + self.training_bias
            self.target_output_layer = tf.matmul(self.target_input_layer, tf.transpose(self.target_weight)) + self.target_bias
           
            if not layer_params.skip_relu:
                self.training_output_layer = tf.nn.relu(self.training_output_layer)
                self.target_output_layer = tf.nn.relu(self.target_output_layer)
            self.layer_description += "relu {} + {}    {}\n".format(weight_size, bias_size, layer_params.name)

class DuelLayersParams:
    def __init__(self):
        self.value_layers = []
        self.advantage_layers = []

class Training_Params:
    def __init__(self):
        self.initial_random_chance = 1.
        self.random_decay = 0.9925
        self.discount_rate = 0.95
        self.memory_size = 10000
        self.train_freq = 8
        self.batch_size = 128
        self.update_param_freq = 32
        self.max_episode = 1000
        self.max_step = 199
        self.sess = None
        self.show_display = True
        self.show_freq = 10

class Neural_Network:
    def __init__(self, env):
        print "NN"
        self.env = env
        
        # this setup currently targets environments with a 
        # Box observation_space and Discrete actions_space
        #self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.training_input_layer = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
        self.target_input_layer = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
        
        self.training_output_layer = self.training_input_layer
        self.target_output_layer = self.target_input_layer


    def _build_layers(self, layer_list, layer_param_list):
        # build network
        for layer_params in layer_param_list:
            if isinstance(layer_params, ConvolutionalLayerParams):
                layer = ConvolutionalLayer(layer_list[-1], layer_params)
                self.layer_model_list.append(layer)
                layer_list.append(layer.get_output_layer())
                self.description += layer.get_description()
            elif isinstance(layer_params, RELULayerParams):
                layer = RELULayer(layer_list[-1], layer_params)
                self.layer_model_list.append(layer)
                layer_list.append(layer.get_output_layer())
                self.description += layer.get_description()
            elif isinstance(layer_params, DuelLayersParams):
                prev_layer = layer_list[-1]

                self.description += "Value Network \n"
                value_list = self._build_layers(
                        [prev_layer], layer_params.value_layers)

                self.description += "Advantage Network \n"
                advantage_list = self._build_layers(
                        [prev_layer], layer_params.advantage_layers)

                self.description += "Value + Avgerage Advantage\n"
                training_output_layer = value_list[-1][0] + (advantage_list[-1][0] - 
                  tf.reduce_mean(advantage_list[-1][0], reduction_indices=1, keep_dims=True))
                target_output_layer = value_list[-1][1] + (advantage_list[-1][1] - 
                  tf.reduce_mean(advantage_list[-1][1], reduction_indices=1, keep_dims=True))
                layer_list.append([training_output_layer, target_output_layer])


        return layer_list

    def train(self, training_params):
        score = 0

        if not training_params.sess:
            training_params.sess = tf.Session()

        self.sess = training_params.sess

        # save information to tensorboard
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('score', score)
        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(".", self.sess.graph)
        
        self.sess.run(tf.initialize_all_variables())

        self.random_chance = training_params.initial_random_chance
        self.memory = []
        self.tick = 0;
        for i_episode in range(training_params.max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(training_params.max_step):
                self.tick += 1
                if training_params.show_display and i_episode % training_params.show_freq == 0:
                    self.env.render()
                output = self.sess.run(self.training_output_layer, feed_dict = {self.training_input_layer: [obs]})
                
                # start off with mostly random actions
                # slowly take away the random actions
                #if random.random() > self.random_chance:
                if np.random.random() < self.random_chance:
                    action = self.env.action_space.sample()
                else: 
                    action = output.argmax()
                  
                newobs, reward, done, info = self.env.step(action)

                score += reward
                self.memory.append([obs, action, reward, newobs, done])
                if len(self.memory) > training_params.memory_size:
                    self.memory.pop(0)

                if done:
                    break
                
                # update environment
                obs = newobs

                if self.tick % training_params.train_freq == 0:
                    self.minibatch(training_params)
           
                if self.tick % training_params.update_param_freq == 0:
                    for layer in self.layer_model_list:
                        layer.update_target(self.sess)

            # slowly take away the random exploration
            self.random_chance *= training_params.random_decay

            # how did we do?
            print "Episode ", i_episode, "\tScore ", score, "\tRandom ", self.random_chance

    def minibatch(self, training_params):
        # learn a batch of runs
        batch_size = training_params.batch_size
        selection = random.sample(self.memory, min(batch_size, len(self.memory)))

        inputs = np.array([r[0] for r in selection])
        
        newobs_list = np.array([r[3] for r in selection])
        next_q_value = self.sess.run(self.target_output_layer, feed_dict = {self.target_input_layer: newobs_list}).max(axis=1)
        
        chosen_q_mask = np.zeros((len(selection), self.output_size))
        target_q = np.zeros((len(selection)))
        for i, run in enumerate(selection):
            obs, action, reward, newobs, done = run
            
            chosen_q_mask[i][action] = 1.
            
            # Q learning update step 
            # Q_now = reward + (discout * future_Q)
            target_q[i] = reward 
            if not done:
                # no future Q if action was terminal
                target_q[i] += (
                        training_params.discount_rate
                        * next_q_value[i])
        
        #print inputs
        _, summary = self.sess.run([self.train_node, self.summary], 
                feed_dict={
                    self.training_input_layer: inputs, 
                    self.target_q_placeholder: target_q, 
                    self.chosen_q_mask_placeholder: chosen_q_mask})
        
        if self.tick % 100 == 0:
            self.summary_writer.add_summary(summary, self.tick)
        

class DuelDualQ(Neural_Network):
    def __init__(self, env, layer_param_list):
        Neural_Network.__init__(self, env)
        self.layer_list = [[self.training_input_layer, self.target_input_layer]]
        self.layer_model_list = []
            
        self.description = ""
            
        layer_param_list.append(RELULayerParams(self.output_size, skip_relu=True, name="output_layer"))
            
        self.layer_list = self._build_layers(self.layer_list, layer_param_list)

        self.training_output_layer, self.target_output_layer = self.layer_list[-1]
            
        print self.description
        with tf.name_scope('Q_learning'):
            self.target_q_placeholder = tf.placeholder(tf.float32, [None])
            self.chosen_q_mask_placeholder = tf.placeholder(tf.float32, [None, self.output_size])
            
            # only train on the q of the action that was taken
            chosen_q = tf.reduce_sum(
                    tf.mul(self.training_output_layer, self.chosen_q_mask_placeholder), 
                    reduction_indices=[1])
            
            self.loss = tf.reduce_mean(tf.square(self.target_q_placeholder  - chosen_q))

        #with tf.name_scope('Weight_minimization'):
        #    for w in self.weight_list:
        #        self.loss += 0.001 * tf.reduce_sum(tf.square(w))
        
        #Adam has a built in learnig rate decay
        self.train_node = tf.train.AdamOptimizer(
                1e-3).minimize(self.loss)


def main():
    #game = "Pendulum-v0"
    #game = "MountainCar-v0"
    game = "CartPole-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/cartpole-experiment-1')

    layer_param_list = []

    layer_param_list.append(
        RELULayerParams(100, name="hl1"))

    value_param_list = []
    
    value_param_list.append(
        RELULayerParams(10, name="value1"))
    
    advantage_param_list = []

    advantage_param_list.append(
            RELULayerParams(100, name="adv1"))
    
    duel_layer_params = DuelLayersParams()
    duel_layer_params.value_layers = value_param_list
    duel_layer_params.advantage_layers = advantage_param_list
    
    layer_param_list.append(duel_layer_params)

    nn = DuelDualQ(env, layer_param_list)

    training_params = Training_Params()

    nn.train(training_params)
    #env.monitor.close()


if __name__ == "__main__":
    main()
