import gym

import tensorflow as tf
import numpy as np

import random

class NeuralNetwork:

    BIAS_INIT = 1e-3
    RANDOM_CHANCE = 0.9
    RANDOM_DECAY = 0.99

    def __init__(self, env, hidden_layer_list, 
            learning_rate=1e-3, 
            discount_rate=0.7):
        self.LEARNING_RATE = np.float32(learning_rate)
        self.DISCOUNT_RATE = np.float32(discount_rate)
        self.env = env
        
        print self.env.observation_space
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.input_layer = tf.placeholder(tf.float32, [None, self.input_size])
        self.layer_list = [self.input_layer]
        self.weight_list = []
        
        # build network
        layer_map = [self.input_size] + hidden_layer_list + [self.output_size]
        for layer_number, layer_size in enumerate(layer_map[1:], start=1):
            with tf.name_scope('layer'+ str(layer_number)):
                previous_size = layer_map[layer_number - 1]
                
                weight = tf.Variable(tf.truncated_normal([layer_size, previous_size], stddev=0.1), name=("weights" + str(layer_number)))
                bias = tf.Variable(tf.constant(self.BIAS_INIT,shape=[layer_size]), name=("bias" + str(layer_number)))

                self.layer_list.append(tf.matmul(self.layer_list[-1], tf.transpose(weight)) + bias)

                if layer_number != len(layer_map) - 1:
                    self.layer_list[-1] = tf.nn.relu(self.layer_list[-1], name = ("layernode"+ str(layer_number)))
                else:
                    self.Q = self.layer_list[-1]

                self.weight_list.append(weight)

        
        with tf.name_scope('Q_learning'):
            self.target_q_placeholder = tf.placeholder(tf.float32, [None])
            self.chosen_q_mask_placeholder = tf.placeholder(tf.float32, [None, self.output_size])

            # only train on the q of the action that was taken
            chosen_q = tf.reduce_sum(tf.mul(self.Q, self.chosen_q_mask_placeholder), reduction_indices=[1])
            
            self.loss = tf.reduce_mean(tf.square(self.target_q_placeholder  - chosen_q))

        #with tf.name_scope('Weight_minimization'):
        #    for w in self.weight_list:
        #        self.loss += 0.001 * tf.reduce_sum(tf.square(w))

        self.train_node = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        
    def train(self, 
            max_episode = 500, 
            memory_size = 10000, 
            show_display = True):
        score = 0
        
        sess = tf.Session()
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('score', score)
        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(".", sess.graph)
        
        sess.run(tf.initialize_all_variables())

        memory = []
        score_list=[]
        for i_episode in range(max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(200):
                if show_display:
                    self.env.render()
                output = sess.run(self.layer_list[-1], feed_dict = {self.input_layer: [obs]})
                
                # start off with mostly random actions
                if random.random() > self.RANDOM_CHANCE:
                    action = output.argmax()
                else: 
                    action = self.env.action_space.sample()
                 
                # slowly take away the random exploration
                self.RANDOM_CHANCE *= self.RANDOM_DECAY
                
                newobs, reward, done, info = self.env.step(action)

                score += reward
                memory.append([obs, action, reward, newobs, done])
                if len(memory) > memory_size:
                    memory.pop(0)

                if done:
                    break
                
                # update environment
                obs = newobs
           
            score_list.append(score)

            # learn a batch of runs
            selection = random.sample(memory, min(200, len(memory)))

            inputs = np.array([r[0] for r in selection])
            
            newobs_list = np.array([r[3] for r in selection])
            next_q_value = sess.run(self.Q, feed_dict = {self.input_layer: newobs_list}).max(axis=1)
            
            chosen_q_mask = np.zeros((len(selection), self.output_size))
            target_q = np.zeros((len(selection)))
            for i, run in enumerate(selection):
                obs, action, reward, newobs, done = run
                
                chosen_q_mask[i][action] = 1.
                
                # Q learning update step 
                # reward + (discout * future_Q)
                target_q[i] = reward 
                if not done:
                    # no future Q if action was terminal
                    target_q[i] += (self.DISCOUNT_RATE * next_q_value[i])
            
            #print inputs
            _, summary = sess.run([self.train_node, self.summary], 
                    feed_dict={
                        self.input_layer: inputs, 
                        self.target_q_placeholder: target_q, 
                        self.chosen_q_mask_placeholder: chosen_q_mask})
            
            
            if i_episode % 10 == 0:
                self.summary_writer.add_summary(summary, i_episode)
            
                # how did we do?
                #print "Episode ", i_episode, "\tScore ", score

        return score_list

def main():
    #game = "Acrobot-v0"
    #game = "MountainCar-v0"
    game = "CartPole-v0"
    #game = "Copy-v0"
    env = gym.make(game)
    data = [["id","learning rate", "discount rate", "info", "average"]]
    for learning_rate_exponent in np.linspace(-4, -1, 15):
        for discount_rate in np.linspace(.5, 1, 15):
            hidden_layer_list = [8]
            nn = NeuralNetwork( env, 
                    hidden_layer_list, 
                learning_rate=10**learning_rate_exponent,
                discount_rate=discount_rate)
            score_list = nn.train(show_display=False)
            print "learning_rate {}\t\
                    discount rate {}\t\
                    max {}\t\
                    average {}".format(
                        10**learning_rate_exponent,
                        discount_rate,                  
                        max(score_list), 
                        np.mean(score_list))
            print score_list[::10]
            data.append(
                    ["", learning_rate_exponent, 
                    discount_rate, "",
                    np.mean(score_list[200:])])
            tf.reset_default_graph()

    print data

if __name__ == "__main__":
    main()
