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
        'INITIAL_RANDOM_CHANCE': 0.9,
        'RANDOM_DECAY': 0.95,
        'LEARNING_RATE': 1e-2,
        'DISCOUNT_RATE': 0.95,
        'MEMORY_SIZE': 50000,
        'BATCH_SIZE': 800
        }

    network_params = {
        'HIDDEN_LAYER_LIST': [100, 100],
        'INITIAL_BIAS': 1e-3
        }

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
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.input_layer = tf.placeholder(tf.float32, [None, self.input_size])
        self.layer_list = [self.input_layer]
        self.weight_list = []
        
        # build network
        layer_map = [self.input_size] + \
            self.network_params['HIDDEN_LAYER_LIST'] + [self.output_size]
        for layer_number, layer_size in enumerate(layer_map[1:], start=1):
            with tf.name_scope('layer'+ str(layer_number)):
                previous_size = layer_map[layer_number - 1]
                
                weight = tf.Variable(tf.truncated_normal(
                    [layer_size, previous_size], stddev=0.1), 
                    name=("weights" + str(layer_number)))
                bias = tf.Variable(tf.constant(
                    self.network_params['INITIAL_BIAS'],
                    shape=[layer_size]), 
                    name=("bias" + str(layer_number)))

                self.layer_list.append(tf.matmul(
                    self.layer_list[-1], tf.transpose(weight)) + bias)

                if layer_number != len(layer_map) - 1:
                    self.layer_list[-1] = tf.nn.relu(self.layer_list[-1],
                            name = ("layernode"+ str(layer_number)))
                else:
                    self.Q = self.layer_list[-1]

                self.weight_list.append(weight)

        with tf.name_scope('Q_learning'):
            self.target_q_placeholder = tf.placeholder(tf.float32, [None])
            self.chosen_q_mask_placeholder = tf.placeholder(tf.float32, [None, self.output_size])

            # only train on the q of the action that was taken
            chosen_q = tf.reduce_sum(
                    tf.mul(self.Q, self.chosen_q_mask_placeholder), 
                    reduction_indices=[1])
            
            self.loss = tf.reduce_mean(tf.square(self.target_q_placeholder  - chosen_q))

        with tf.name_scope('Weight_minimization'):
            for w in self.weight_list:
                self.loss += 0.001 * tf.reduce_sum(tf.square(w))

        #Adam has a built in learnig rate decay
        self.train_node = tf.train.AdamOptimizer(
                self.training_params['LEARNING_RATE']).minimize(self.loss)
        
    def train(self, 
            max_episode = 1000,
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
        memory = []
        for i_episode in range(max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(199):
                if show_display and i_episode % 10 == 0:
                    self.env.render()
                output = sess.run(self.layer_list[-1], feed_dict = {self.input_layer: [obs]})
                
                # start off with mostly random actions
                # slowly take away the random actions
                if random.random() > self.random_chance:
                    action = output.argmax()
                else: 
                    action = self.env.action_space.sample()
                  
                newobs, reward, done, info = self.env.step(action)

                score += reward
                memory.append([obs, action, reward, newobs, done])
                if len(memory) > self.training_params['MEMORY_SIZE']:
                    memory.pop(0)

                if done:
                    break
                
                # update environment
                obs = newobs
           
            # slowly take away the random exploration
            self.random_chance *= self.training_params['RANDOM_DECAY']


            # learn a batch of runs
            batch_size = self.training_params['BATCH_SIZE']
            selection = random.sample(memory, min(batch_size, len(memory)))

            inputs = np.array([r[0] for r in selection])
            
            newobs_list = np.array([r[3] for r in selection])
            next_q_value = sess.run(self.Q, feed_dict = {self.input_layer: newobs_list}).max(axis=1)
            
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
                            self.training_params['DISCOUNT_RATE']
                            * next_q_value[i])
            
            #print inputs
            _, summary = sess.run([self.train_node, self.summary], 
                    feed_dict={
                        self.input_layer: inputs, 
                        self.target_q_placeholder: target_q, 
                        self.chosen_q_mask_placeholder: chosen_q_mask})
            
            if i_episode % 10 == 0:
                self.summary_writer.add_summary(summary, i_episode)
            
                # how did we do?
                print "Episode ", i_episode, "\tScore ", score

def main():
    #game = "Acrobot-v0"
    #game = "MountainCar-v0"
    game = "CartPole-v0"
    hidden_layer_list = [100,100]
    #env = gym.make(game)
    env.monitor.start('/tmp/cartpole-experiment-1')
    nn = NeuralNetwork(env)
    nn.train(show_display=True)
    #env.monitor.close()

if __name__ == "__main__":
    main()
