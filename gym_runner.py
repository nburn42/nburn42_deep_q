"""
Nathan Burnham GymActor is a generic framework for running different types of actors on gym environments
"""
import gym

import tensorflow as tf
import numpy as np

import random


class Training_Params:
    def __init__(self):
        self.initial_random_chance = 1.
        self.max_episode = 1000
        self.max_step = 199
        self.sess = None
        self.show_display = True
        self.show_freq = 10
        self.random_decay = 0.9925

class Gym_Runner:
    def __init__(self, env, actor):
        self.actor = actor
        self.env = env


    def train(self, training_params):
        score = 0

        if not training_params.sess:
            training_params.sess = tf.Session()

        self.sess = training_params.sess

        self.actor.build(self.sess)



        # todo move random chance to actors
        self.random_chance = training_params.initial_random_chance
        self.tick = 0;
        for i_episode in range(training_params.max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(training_params.max_step):
                self.tick += 1
                if training_params.show_display and i_episode % training_params.show_freq == 0:
                    self.env.render()
                
                
                # start off with mostly random actions
                # slowly take away the random actions
                #if random.random() > self.random_chance:
                if np.random.random() < self.random_chance:
                    action = self.env.action_space.sample()
                else: 
                    action = self.actor.get_action(obs)
                  
                # slowly take away the random exploration
                self.random_chance *= training_params.random_decay

                newobs, reward, done, info = self.env.step(action)

                score += reward
                self.actor.step([obs, action, reward, newobs, done])

                if done:
                    break
                
                # update environment
                obs = newobs

            # how did we do?
            print "Episode ", i_episode, "\tScore ", score, "\tRandom ", self.random_chance

