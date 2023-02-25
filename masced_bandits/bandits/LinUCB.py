from masced_bandits.bandits.Bandit import Bandit
from masced_bandits.bandit_options import bandit_args

import numpy as np


FORMULA_FUNC = None

CUM_REWARD = 0
CUM_SQ_REWARD = 1
N_K = 2


class LinUCB(Bandit):
    def __init__(self, **kwargs):
        feature_len = int(kwargs.get("feature_len",""))
        super().__init__("LinUCB-" + str(feature_len))
        
        self.alpha = float(kwargs.get("alpha", 0.1))
        try:
            self.prev_features = bandit_args['initial_features']
        except:
            print("When using LinUCB you need to specify the features that accompanied the initial reward. This initial reward reflects the state of the system before using a bandit.")
            raise RuntimeError("No initial features specified")



        self.bandit_round = -1
        self.arm_matrices_A = {}
        self.arm_vectors_b = {}

        for arm in self.arms: #this operation can also be done if we want to have a varying amount of arms, to add new arms
            self.arm_matrices_A[arm]= np.identity(feature_len)
            self.arm_vectors_b[arm]= np.zeros(feature_len)

        


        
    def get_next_arm(self, reward, features):
        #Here I receive the reward for the previous arm, and the current context features
        self.bandit_round+=1
        #update
        self.arm_matrices_A[self.last_action]+= (self.prev_features @ self.prev_features.T)
        self.arm_vectors_b[self.last_action]+= (reward*self.prev_features)#scalar multiplication on a vector so no @.



        next_arm = max(self.arms, key=lambda arm: self.some_operation(arm,features))

        self.last_action = next_arm
        self.prev_features = features
        return next_arm


    def some_operation(self, arm, features):
        inverse_of_a = np.linalg.inv(self.arm_matrices_A[arm])
        theta = inverse_of_a @ self.arm_vectors_b[arm] #matmul
        p = (theta.T @ features) + self.alpha * np.sqrt(features.T @ inverse_of_a @ features)
        return p

    def reward_average(self, arm):
        return self.arm_reward_pairs[arm][CUM_REWARD] / self.arm_reward_pairs[arm][N_K]

