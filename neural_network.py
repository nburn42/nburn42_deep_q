"""
Nathan Burnham Tensorflow Neural Network library
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

    def get_weight_image_summary(self):
        return (tf.image_summary(self.name + "w", self._format_weight(self.training_weight), max_images=100), 
                tf.image_summary(self.name + "b", self._format_weight(self.training_bias), max_images=100))

    def _format_weight(self, tensor):
        newtensor = tensor
        if(newtensor.get_shape().ndims == 1):
            newtensor = tf.expand_dims(tensor, 0)
        if(newtensor.get_shape().ndims < 3):
            newtensor = tf.expand_dims(tensor,0)
        while(newtensor.get_shape().ndims < 4):
            newtensor = tf.expand_dims(newtensor, -1)
        print self.name
        print newtensor.get_shape()
        return newtensor

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
        self.name = layer_params.name
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
        self.name = layer_params.name
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


class Deep_Q_Params:
    def __init__(self):
        self.env = None
        self.layer_param_list = None
        self.memory_size = 10000
        self.discount_rate = 0.95
        self.memory_size = 10000
        self.train_freq = 8
        self.batch_size = 128
        self.update_param_freq = 128

class Deep_Q:
    def __init__(self, deep_q_params):
        #todo don't need to break out these variables
        self.env = deep_q_params.env
        self.layer_param_list = deep_q_params.layer_param_list
        self.params = deep_q_params


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

    def build(self, sess):
        self.sess = sess
        # this setup currently targets environments with a 
        # Box observation_space and Discrete actions_space
        #self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.training_input_layer = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
        self.target_input_layer = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
        
        self.training_output_layer = self.training_input_layer
        self.target_output_layer = self.target_input_layer
        
        
        self.layer_list = [[self.training_input_layer, self.target_input_layer]]
        self.layer_model_list = []
        layer_param_list = self.layer_param_list

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

        self.memory = []
        self.tick = 0


        tf.scalar_summary("loss", self.loss)
        for layer in self.layer_model_list:
            layer.get_weight_image_summary()

        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('deepq', self.sess.graph)
        sess.run(tf.initialize_all_variables())

    def get_action(self, obs):
        output = self.sess.run(self.training_output_layer, feed_dict = {self.training_input_layer: [obs]})
        

        return output.argmax()

    def step(self, state):
        self.memory.append(state)
        if len(self.memory) > self.params.memory_size:
            self.memory.pop(0)

        if self.tick % self.params.train_freq == 0:
            self.minibatch()
           
        if self.tick % self.params.update_param_freq == 0:
            for layer in self.layer_model_list:
                layer.update_target(self.sess)

        self.tick += 1


    def minibatch(self):
        # learn a batch of runs
        batch_size = self.params.batch_size
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
                        self.params.discount_rate
                        * next_q_value[i])
        
        #print inputs
        _, summary = self.sess.run([self.train_node, self.summary], 
                feed_dict={
                    self.training_input_layer: inputs, 
                    self.target_q_placeholder: target_q, 
                    self.chosen_q_mask_placeholder: chosen_q_mask})
        
        if self.tick % 100 == 0:
            self.summary_writer.add_summary(summary, self.tick)

