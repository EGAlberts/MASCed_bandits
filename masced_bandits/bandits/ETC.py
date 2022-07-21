
import numpy as np
import time
from masced_bandits.utilities import calculate_utility, convert_conf
from masced_bandits.bandit_options import bandit_args
from masced_bandits.bandits.Bandit import Bandit



REWARD = 0
ACTION = 1
class ETC(Bandit):
    def __init__(self, **kwargs):
        super().__init__("ETC")
        #self.formula = self.formula_to_function(formula)

        initial_configuration = bandit_args["initial_configuration"]
        self.bandit_round = 0
        self.game_list = []
        self.last_action = initial_configuration
        self.explore_factor = float(kwargs['exploration_rounds'])

    def start_strategy(self, reward):
        self.bandit_round = self.bandit_round + 1

        self.game_list.append([reward, self.last_action])


        if  self.bandit_round <= ((len(self.arms)-1) * self.explore_factor):
            new_action = self.arms[(self.bandit_round % len(self.arms))]
        else: 
            new_action = max(self.arms, key=lambda arm: self.reward_average(arm))



        self.last_action = new_action
        return new_action


    def reward_average(self, arm):
        r_sum = 0

        for game in self.game_list:  
            if(game[ACTION] == arm): #the games in which arm was chosen
                r_sum+=game[REWARD]
     
        times_arm_played = self.times_played(arm) 
        if(r_sum == 0 or times_arm_played == 0): return 0
            
        return r_sum/times_arm_played
    
    def times_played(self, arm):
        return len([game for game in self.game_list if game[ACTION] == arm])




    

    


