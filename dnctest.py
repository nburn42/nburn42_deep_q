# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random
import bisect
import nplot

# HYPERPARMETERS
H = 150
H2 = 150
batch_number = 500
gamma = 0.995
num_of_ticks_between_q_copies = 1000
explore_decay = 0.99999
min_explore = 0.05
max_steps = 499    
max_episodes = 1500
memory_size = 20000
learning_rate = 1e-3
    

if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
   
    print env.observation_space
    print env.action_space

    inputsize = env.observation_space.shape[0]
    outputsize = env.action_space.n

    tf.reset_default_graph()

    input_width = 10
    hidden_width = 10
    lstm_width = input_width + hidden_width + hidden_width 

    lstm_size = 50
    lstm_layer_count = 3

    lstm_layers = []

    # start layer
    x_placeholder = tf.placeholder(tf.float32, [None, input_width])     
    hl_0 = tf.zeros([None, hidden_width])
    ht_0 = tf.zeros([None, hidden_width])
    state = tf.concat(1, [x_placeholder, h_0, h_t])
    print "state size: ", state.get_shape()

    lstm = rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm] * lstm_layer_count,
                state_is_tuple=False)

    # build lstm
    for plan in range(lstm_layer_count):
        for 
        wi = tf.Variable(tf.random_uniform([inputsize,lstm_width], -.01, .01))
        bi = tf.Variable(tf.random_uniform([lstm_width], -.01, .01))
        
        wf = tf.Variable(tf.random_uniform([inputsize,lstm_width], -.01, .01))
        bf = tf.Variable(tf.random_uniform([lstm_width], -.01, .01))

        i = tf.sigmoid(tf.matmul(wi,temp_input) + bi)
        
    
    #First Q Network
    w1 = tf.Variable(tf.random_uniform([inputsize,H], -.10, .10))
    bias1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    
    w2 = tf.Variable(tf.random_uniform([H, H2], -.10, .10))
    bias2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    
    w3 = tf.Variable(tf.random_uniform([H2, outputsize], -.10, .10))
    bias3 = tf.Variable(tf.random_uniform([outputsize], -.10, .10))
    
    w1_prime = tf.Variable(tf.random_uniform([inputsize,H], -1.0, 1.0))
    bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    w3_prime = tf.Variable(tf.random_uniform([H2, outputsize], -1, 1))
    bias3_prime = tf.Variable(tf.random_uniform([outputsize], -1, 1))
    
    #Make assign functions for updating Q prime's weights
    w1_prime_update= w1_prime.assign(w1)
    bias1_prime_update= bias1_prime.assign(bias1)
    w2_prime_update= w2_prime.assign(w2)
    bias2_prime_update= bias2_prime.assign(bias2)
    w3_prime_update= w3_prime.assign(w3)
    bias3_prime_update= bias3_prime.assign(bias3)
   
    all_assigns = [
            w1_prime_update, 
            w2_prime_update, 
            w3_prime_update, 
            bias1_prime_update, 
            bias2_prime_update, 
            bias3_prime_update]


    #build network
    states_placeholder = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
    hidden_1 = tf.nn.relu(tf.matmul(states_placeholder, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    hidden_2 = tf.nn.dropout(hidden_2, .5)
    Q = tf.matmul(hidden_2, w3) + bias3

    hidden_1_prime = tf.nn.relu(tf.matmul(states_placeholder, w1_prime) + bias1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    hidden_2_prime = tf.nn.dropout(hidden_2_prime, .5)
    Q_prime =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime

    action_used_placeholder = tf.placeholder(tf.int32, [None], name="action_masks") 
    action_masks = tf.one_hot(action_used_placeholder, outputsize)
    filtered_Q = tf.reduce_sum(tf.mul(Q, action_masks), reduction_indices=1) 
    
    #we need to train Q
    target_q_placeholder = tf.placeholder(tf.float32, [None,]) # This holds all the rewards that are real/enhanced with Qprime
    loss = tf.reduce_sum(tf.square(filtered_Q - target_q_placeholder))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment

    D = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    xmax = 1
    ymax = 1
    xind = 1
    yind = 1

    init = tf.initialize_all_variables()
  
    
    with tf.Session() as sess:
        sess.run(init)
        sess.run(all_assigns)
        
        ticks = 0
        for episode in xrange(max_episodes):
            state = env.reset()
            
            reward_sum = 0
            
            for step in xrange(max_steps):
                ticks += 1
                
                #print state
                xmax = max(xmax, state[xind])
                ymax = max(ymax, state[yind])

                if episode % 10 == 0:
                    q, qp = sess.run([Q,Q_prime], feed_dict={states_placeholder: np.array([state])})
                    print "Q:{}, Q_ {}".format(q[0], qp[0])
                    #print "T: {} S {}".format(ticks, state)
                    env.render()

                if explore > random.random():
                    action = env.action_space.sample()
                else:
                    #get action from policy
                    q = sess.run(Q, feed_dict={states_placeholder: np.array([state])})[0]
                    action = np.argmax(q)
                explore = max(explore * explore_decay, min_explore)
                
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward
                #print reward

                D.append([state, action, reward, new_state, done])
                if len(D) > memory_size:
                    D.pop(0);
           
                state = new_state

               
                if done: 
                    break
                
                #Training a Batch
                samples = random.sample(D, min(batch_number, len(D)))

                #print samples

                #calculate all next Q's together for speed
                new_states = [ x[3] for x in samples]
                all_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: new_states})

                y_ = []
                state_samples = []
                actions = []
                terminalcount = 0
                for ind, i_sample in enumerate(samples):
                    state_mem, curr_action, reward, new_state, done = i_sample
                    if done:
                        y_.append(reward)
                        terminalcount += 1
                    else:
                        #this_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: [new_state]})[0]
                        this_q_prime = all_q_prime[ind]
                        maxq = max(this_q_prime)
                        y_.append(reward + (gamma * maxq))

                    state_samples.append(state_mem)

                    actions.append(curr_action);
                sess.run([train], feed_dict={states_placeholder: state_samples, target_q_placeholder: y_, action_used_placeholder: actions})
                if ticks % num_of_ticks_between_q_copies == 0:
                    sess.run(all_assigns)
                    
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
        if True:#episode % 30 == 0:
                        teststate = [0 for x in xrange(env.observation_space.shape[0])]
                        #print "S: ", teststate
                        X=[]
                        Y=[]
                        Z=[]
                        ZR=[]
                       
                        xmin = -xmax
                        xstep = xmax/100.0

                        ymin = -ymax
                        ystep = ymax/100.0

                        test_state_list = []
                        for x in nplot.drange(xmin,xmax, xstep):
                            for y in nplot.drange(ymin,ymax,ystep):
                                teststate[xind] = x
                                teststate[yind] = y
                                test_state_list.append([teststate[x] for x in xrange(len(teststate))])

                        test_q_list = sess.run(Q, feed_dict={states_placeholder:test_state_list})
                        zmax = max(map(max,test_q_list))
                        ind = 0
                        for x in nplot.drange(xmin,xmax, xstep):
                            XX = []
                            YY = []
                            ZZ = []
                            ZZR = []
                            for y in nplot.drange(ymin,ymax,ystep):
                                XX.append(x)
                                YY.append(y)
                                ZZ.append(test_q_list[ind][0])
                                ZZR.append(test_q_list[ind][1])
                                ind += 1
                            X.append(XX)
                            Y.append(YY)
                            Z.append(ZZ)
                            ZR.append(ZZR)
                        nplot.plot(X,Y,Z, ZR, xmin,ymax,zmax)


                
            
            
                
                
    env.monitor.close()
