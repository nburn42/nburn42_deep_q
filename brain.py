# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random
import bisect

class SuperLayer:
    state_size = 10
    input_action_size = 3
    loss_size = 1
    #value_size = 1
    rep_buleprint = [5]
    model_buleprint = [5, 5]
    actor_blueprint = [5, 5]
    q_blueprint = [5, 5]
    attention_blueprint = [5]
    #goal_attention_blueprint = [10]
    #state_memory_size = []
    loss = {}
    q_loss = {}
    assign = []
    

# HYPERPARMETERS

batch_number = 5#100
gamma = 0.995
num_of_ticks_between_q_copies = 1000
explore_decay = 0.99999
min_explore = 0.05
max_steps = 499    
max_episodes = 1500
memory_size = 20000
learning_rate = 1e-3
    

if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
   
    print env.observation_space
    print env.action_space

    input_state_size = env.observation_space.shape[0]
    output_action_size = env.action_space.shape[0]

    tf.reset_default_graph()

    #super layer blueprints
    super_layer_list = []
    sl = SuperLayer()
    sl.output_action_size = output_action_size
    super_layer_list.append(sl)
    sl = SuperLayer()
    sl.output_action_size = 3
    sl.input_action_size = 3
    super_layer_list.append(sl)


    # start layer
    raw_state_placeholder = tf.placeholder(tf.float32, [None, input_state_size], name="raw_state") 
    first_state = last_state = last_state_ = raw_state_placeholder

    for index, sl in enumerate(super_layer_list):
        sl.last_action = tf.placeholder(tf.float32, [None, sl.output_action_size], name="last_action_{}".format(index))
        sl.last_loss = tf.placeholder(tf.float32, [None, sl.loss_size], name="last_loss_{}".format(index))
        last_state = tf.concat(1, [last_state, sl.last_action, sl.last_loss])
        last_state_ = tf.concat(1, [last_state_, sl.last_action, sl.last_loss])

    for index, sl in enumerate(super_layer_list):
        last_layer = last_state
        last_layer_ = last_state_

        sl.attention_blueprint.append(last_layer.get_shape()[1].value)
        for lb in sl.attention_blueprint:
            w = tf.Variable(tf.random_uniform([last_layer.get_shape()[1].value ,lb], -.01, .01))
            b = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)          

            w_ = tf.Variable(tf.random_uniform([last_layer_.get_shape()[1].value ,lb], -.01, .01))
            b_ = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer_ = tf.nn.relu(tf.matmul(last_layer_, w_) + b_)        

            #todo replace with gated assign
            sl.assign.append(w_.assign(w))
            sl.assign.append(b_.assign(b))
        
        sl.attention = last_layer
        sl.attention_ = last_layer_
        
        sl.last_rep_pred_placeholder = tf.placeholder(tf.float32, [None,sl.rep_buleprint[-1]])
        last_layer = tf.mul(last_state, tf.nn.softmax(sl.attention))
        last_layer_ = tf.mul(last_state_, tf.nn.softmax(sl.attention_))
        
        for lb in sl.rep_buleprint:
            w = tf.Variable(tf.random_uniform([last_layer.get_shape()[1].value ,lb], -.01, .01))
            b = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)          

            w_ = tf.Variable(tf.random_uniform([last_layer_.get_shape()[1].value ,lb], -.01, .01))
            b_ = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer_ = tf.nn.relu(tf.matmul(last_layer_, w_) + b_)        
            #todo replace with gated assign
            sl.assign.append(w_.assign(w))
            sl.assign.append(b_.assign(b))
        
        sl.rep = last_layer
        sl.rep_ = last_layer_
        
        sl.model_buleprint.append(sl.rep.get_shape()[1].value)

        for lb in sl.model_buleprint:
            w = tf.Variable(tf.random_uniform([last_layer.get_shape()[1].value ,lb], -.01, .01))
            b = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)          

            w_ = tf.Variable(tf.random_uniform([last_layer_.get_shape()[1].value ,lb], -.01, .01))
            b_ = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer_ = tf.nn.relu(tf.matmul(last_layer_, w_) + b_)        
            #todo replace with gated assign
            sl.assign.append(w_.assign(w))
            sl.assign.append(b_.assign(b))

        sl.rep_pred = last_layer
        sl.rep_pred_ = last_layer_
        sl.loss['rep_pred'] = tf.nn.l2_loss(sl.rep - sl.last_rep_pred_placeholder)
        sl.train = tf.train.AdamOptimizer(0.001).minimize(sl.loss['rep_pred']) 
        
        last_state = tf.nn.dropout(tf.concat(1, [last_state, sl.rep, sl.rep_pred]), .25)
        last_state_ = tf.concat(1, [last_state_, sl.rep_, sl.rep_pred_])

        #last action feeds itself
        next_action = super_layer_list[
                min(index + 1, len(super_layer_list)-1)].last_action
        last_layer = tf.concat(1, [sl.rep, sl.rep_pred, next_action])
        last_layer_ = tf.concat(1, [sl.rep_, sl.rep_pred_, next_action])
   
        sl.actor_blueprint.append(sl.output_action_size)

        for lb in sl.actor_blueprint:
            w = tf.Variable(tf.random_uniform([last_layer.get_shape()[1].value ,lb], -.01, .01))
            b = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)          

            w_ = tf.Variable(tf.random_uniform([last_layer_.get_shape()[1].value ,lb], -.01, .01))
            b_ = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer_ = tf.nn.relu(tf.matmul(last_layer_, w_) + b_)        
            #todo replace with gated assign
            sl.assign.append(w_.assign(w))
            sl.assign.append(b_.assign(b))
     
        sl.explore_placeholder = tf.placeholder(
                tf.float32,[None, sl.output_action_size], name="explore")
        sl.action = tf.tanh(last_layer) #+ sl.explore_placeholder
        sl.action_ = tf.tanh(last_layer_) #+ sl.explore_placeholder

        sl.q_blueprint.append(1)
        
        last_layer = tf.concat(1, [sl.rep, sl.rep_pred, sl.action])
        last_layer_ = tf.concat(1, [sl.rep_, sl.rep_pred_, sl.action_])

        for lb in sl.q_blueprint:
            w = tf.Variable(tf.random_uniform([last_layer.get_shape()[1].value ,lb], -.01, .01))
            b = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer = tf.nn.relu(tf.matmul(last_layer, w) + b)          

            w_ = tf.Variable(tf.random_uniform([last_layer_.get_shape()[1].value ,lb], -.01, .01))
            b_ = tf.Variable(tf.random_uniform([lb], -.01, .01))
            last_layer_ = tf.nn.relu(tf.matmul(last_layer_, w_) + b_)        
            #todo replace with gated assign
            sl.assign.append(w_.assign(w))
            sl.assign.append(b_.assign(b))

        sl.q = last_layer
        sl.q_ = last_layer
    
        sl.target_q_placeholder = tf.placeholder(tf.float32, [None,], name="target_q_{}".format(index)) 
        
        sl.q_loss = tf.reduce_sum(tf.square(sl.q - sl.target_q_placeholder))
        sl.q_train = tf.train.AdamOptimizer(0.001).minimize(sl.q_loss) 
    D = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
  
    feed = []
    q_feed_ = []
    q_train_feed = []
    all_assigns = []
    for i, sl in enumerate(super_layer_list):
        feed.append(sl.action)
        feed.append(sl.loss['rep_pred'])
        feed.append(sl.rep_pred)
        feed.append(sl.train)
        q_feed_.append(sl.q)
        q_train_feed.append(sl.q_train)
        all_assigns.extend(sl.assign)

    with tf.Session() as sess:
        sess.run(init)
        sess.run(all_assigns)
        
        ticks = 0
        for episode in xrange(max_episodes):
            state = env.reset()
            last_action = None
            last_feed_dict = {
                raw_state_placeholder: [state],
                }
            for i, sl in enumerate(super_layer_list):
                last_feed_dict[sl.last_action] = [
                        [0 for x in range(sl.output_action_size)]]
                last_feed_dict[sl.last_loss] = [[0]]
                last_feed_dict[sl.last_rep_pred_placeholder] = [[
                        0 for x in range(sl.rep_buleprint[-1])]]
                last_feed_dict[sl.explore_placeholder] = [[
                    0 for x in range(sl.output_action_size)]]
            reward_sum = 0
    
            for step in xrange(max_steps):
                ticks += 1
                
                if episode % 10 == 0:
                    env.render()

                action = sess.run(feed, feed_dict=last_feed_dict)
                if explore > np.random.uniform():
                    action[0][0] =  env.action_space.sample()
        
                explore *= .9995

                #raw_action = (action[0][0] + env.action_space.low) * (
                #        env.action_space.high - env.action_space.low)
                #print "RA", raw_action
                #print "as", env.action_space.high
                #print "as", env.action_space.low
                new_state, reward, done, _ = env.step(action[0][0])
                reward_sum += reward
                
                state = new_state
               
                feed_dict = {
                        raw_state_placeholder: [state],}
                for i, sl in enumerate(super_layer_list):
                    feed_dict[sl.last_action] = action[4*i]
                    feed_dict[sl.last_loss] = [[
                            action[(4*i)+1],
                            ]]
                    feed_dict[sl.last_rep_pred_placeholder] = action[(4*i)+2]
                    feed_dict[sl.explore_placeholder] = [[
                        np.random.uniform(-explore, explore)
                        for x in range(sl.output_action_size)]]
                D.append([reward, done, last_feed_dict, feed_dict])
                if len(D) > memory_size:
                    D.pop(0);
                
           
                last_feed_dict = feed_dict

               
                if done: 
                    break
                
                #Training a Batch
                samples = random.sample(D, min(batch_number, len(D)))

                sample_last_feed_dict = {} 
                sample_feed_dict = {}
                for key in samples[0][2].keys():
                    sample_last_feed_dict[key] = []
                    sample_feed_dict[key] = []
                for sample in samples:
                    for key in sample[2].keys():
                        sample_last_feed_dict[key].append(sample[2][key][0])
                        sample_feed_dict[key].append(sample[3][key][0])
                for sl in super_layer_list:
                    sample_feed_dict[sl.explore_placeholder] = [[0 for x in range(sl.output_action_size)] for y in range(len(samples))]

                #calculate all next Q's together for speed
                all_q_prime = sess.run(q_feed_, feed_dict=sample_feed_dict)
               
                train_feed_dict = last_feed_dict
                for i, sl in enumerate(super_layer_list):
                    sl.y_ = []
                    train_feed_dict[sl.target_q_placeholder] = sl.y_
                for i, i_sample in enumerate(samples):
                    reward, done, last_feed_dict, feed_dict = i_sample
                    for index, sul in enumerate(super_layer_list):
                        if done:
                            sul.y_.append(reward)
                        else:
                            this_q_prime = all_q_prime[index][i]
                            sul.y_.append(reward + (gamma * this_q_prime[0]))
                for sl in super_layer_list:
                    train_feed_dict[sl.explore_placeholder] = [[
                        0 for x in range(sl.output_action_size)] 
                        for y in range(len(samples))]

                #todo update samples with loss
                sess.run(q_train_feed, feed_dict=train_feed_dict)
                if ticks % num_of_ticks_between_q_copies == 0:
                    sess.run(all_assigns)
                    
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)

    env.monitor.close()
