"""
Deep Q Network
for open ai gym

Written By: Nathan Burnham, nburn42@gmail.com
"""
import gym

import tensorflow as tf
import numpy as np

import random



class NeuralNetwork:

    training_params = {
        'INITIAL_RANDOM_CHANCE': 1.0,
        'RANDOM_DECAY': 0.99,
        'RANDOM_MIN': 0.0,
        'LEARNING_RATE': 1e-2,
        'DISCOUNT_RATE': 0.995,
        'TRAIN_FREQ': 50, #number of steps between training
        'MEMORY_SIZE': 10000,
        'BATCH_SIZE': 800,
        'DISPLAY_FREQ': 10
        }

    network_params = {
        'CONV_LAYER_LIST': [[[8,8],32, 4], [[4,4],64, 2], [[3,3], 64, 1]],
        'RELU_LAYER_LIST': [256, 256],
        'INITIAL_BIAS': 1e-3
        }


    def make_weights(self, size, name="weights"):
        return tf.Variable(tf.truncated_normal(
            shape=size,
            stddev=0.1), 
            name=name)

    def make_bias(self, size, name="bias"):
        return tf.Variable(tf.constant(
            self.network_params['INITIAL_BIAS'],
            shape=size), 
            name=name)

    def make_conv(self, previous_layer, weight, stride):
        return tf.nn.conv2d(previous_layer, weight, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool_2x2(self, previous_layer):
        return tf.nn.max_pool(previous_layer, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self, env, 
            training_params = None, 
            network_params = None):
        self.env = env
        
        if training_params:
            self.training_params = training_params
        if network_params:
            self.network_params = network_params


        # this setup currently targets environments with a 
        # Box observation_space and Discrete actions_space
        #self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.input_layer = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
        self.layer_list = [self.input_layer]
        self.bias_list = []
        self.weight_list = []

        # build network

        #build conv network
        if len(self.layer_list[-1].get_shape()) > 3:
            conv_layer_list = self.network_params["CONV_LAYER_LIST"]
            for layer_number, layer_info in enumerate(conv_layer_list):
                with tf.name_scope('conv_layer' + str(layer_number)):
                    patch_size, features, stride = layer_info
                    conv_depth = int(self.layer_list[-1].get_shape()[3])
                    weight = self.make_weights(patch_size+[conv_depth,features])
                    bias = self.make_bias([features])
                    conv = tf.nn.relu(self.make_conv(self.layer_list[-1], weight, stride) + bias)
                    self.layer_list.append(self.max_pool_2x2(conv))


        #build relu layers
        relu_layer_list = self.network_params["RELU_LAYER_LIST"] + [self.output_size]
        for layer_number, layer_size in enumerate(relu_layer_list):
            with tf.name_scope('relu_layer'+ str(layer_number)):
                #flatten network if not yet flattened
                if len(self.layer_list[-1].get_shape()) > 2:
                    previous_layer = self.layer_list[-1]
                    flat_size = int(reduce(lambda x,y : x*y, previous_layer.get_shape()[1:]))
                    self.layer_list.append(
                            tf.reshape(previous_layer, [-1, flat_size]))

                previous_size = int(self.layer_list[-1].get_shape()[1])
                weight = self.make_weights([layer_size, previous_size], "weights" + str(layer_number))
                bias = self.make_bias([layer_size],("bias" + str(layer_number)))

                self.layer_list.append(tf.matmul(
                    self.layer_list[-1], tf.transpose(weight)) + bias)
                
                self.Q = self.layer_list[-1]

                self.layer_list[-1] = tf.nn.relu(self.layer_list[-1],
                    name = ("layernode"+ str(layer_number)))

                self.weight_list.append(weight)
                self.bias_list.append(bias)


        with tf.name_scope('Q_learning'):
            self.target_q_placeholder = tf.placeholder(tf.float32, [None])
            self.chosen_q_mask_placeholder = tf.placeholder(tf.float32, [None, self.output_size])

            # only train on the q of the action that was taken
            chosen_q = tf.reduce_sum(
                    tf.mul(self.Q, self.chosen_q_mask_placeholder), 
                    reduction_indices=[1])
            
            self.loss = tf.reduce_mean(tf.square(self.target_q_placeholder  - chosen_q))

        #with tf.name_scope('Weight_minimization'):
        #    for w in self.weight_list:
        #        self.loss += 0.001 * tf.reduce_sum(tf.square(w))
        
        #with tf.name_scope('Bias_minimization'):
        #    for b in self.bias_list:
        #        self.loss += 0.001 * tf.reduce_sum(tf.square(b))

        #Adam has a built in learnig rate decay
        self.train_node = tf.train.AdamOptimizer(
                self.training_params['LEARNING_RATE']).minimize(self.loss)
        
    def run(self, 
            max_episode = 1000,
            max_step = 199,
            sess = None,
            show_display = True):
        score = 0

        if not sess:
            sess = tf.Session()

        # save information to tensorboard
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('score', score)
        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(".", sess.graph)
        
        sess.run(tf.initialize_all_variables())

        self.random_chance = self.training_params["INITIAL_RANDOM_CHANCE"]
        self.step_count = 0
        self.memory = []
        for i_episode in range(max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(max_step):
                if show_display and i_episode % self.training_params["DISPLAY_FREQ"] == 0:
                    self.env.render()
                output = sess.run(self.layer_list[-1], feed_dict = {self.input_layer: [obs]})

                # start off with mostly random actions
                # slowly take away the random actions
                #if random.random() > self.random_chance:
                if i_episode < 500 \
                        or np.random.random() < self.random_chance:
                    action = self.env.action_space.sample()
                else: 
                    action = output.argmax()
                  
                newobs, reward, done, info = self.env.step(action)
                #if show_display and i_episode % 10 == 0:
                #    print "obs", newobs
                #    print "reward", reward
                #    print "q",output

                score += reward
                self.memory.append([obs, action, reward, newobs, done])
                if len(self.memory) > self.training_params['MEMORY_SIZE']:
                    self.memory.pop(0)

                if done:
                    break
                
                # update environment
                obs = newobs
           
            # slowly take away the random exploration
            self.random_chance *= self.training_params['RANDOM_DECAY']

            if self.step_count % self.training_params['TRAIN_FREQ'] == 0:
                self.train(sess) 

            if i_episode % self.training_params['DISPLAY_FREQ'] == 0:
                #summary = sess.run(self.summary)
                #self.summary_writer.add_summary(summary, i_episode)
            
                # how did we do?
                print "Episode ", i_episode, "\tScore ", score, "\tRandom ", self.random_chance
    def train(self, sess): 
        # learn a batch of runs
        batch_size = self.training_params['BATCH_SIZE']
        selection = random.sample(self.memory, min(batch_size, len(self.memory)))

        inputs = np.array([r[0] for r in selection])
        
        newobs_list = np.array([r[3] for r in selection])
        test = sess.run(self.Q, 
                feed_dict = {self.input_layer: newobs_list})
        next_q_value = sess.run(self.Q, 
                feed_dict = {self.input_layer: newobs_list}
                ).max(axis=1)
        
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
                target_q[i] += max(-2**29, min(2**29,
                        self.training_params['DISCOUNT_RATE']
                        * next_q_value[i]))
        
        sess.run(self.train_node, 
            feed_dict={
                self.input_layer: inputs, 
                self.target_q_placeholder: target_q, 
                self.chosen_q_mask_placeholder: chosen_q_mask})

def main():
    #game = "Pendulum-v0"
    #game = "MountainCar-v0"
    game = "CartPole-v0"
    env = gym.make(game)
    #env.monitor.start('/tmp/'+ game)
    nn = NeuralNetwork(env)
    nn.run(show_display=True)
    #env.monitor.close()

if __name__ == "__main__":
    main()
