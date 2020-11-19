import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, Buffer_size, Batch_size, Buffer_epsilon, Buffer_alpha):
        self.buffer_size = Buffer_size
        self.batch_size = Batch_size
        self.eps = Buffer_epsilon
        self.buf = collections.deque(maxlen=self.buffer_size)
        self.Weights = collections.deque(maxlen=self.buffer_size)
        self.alpha = Buffer_alpha

    def save_transition(self, transition):
        self.buf.append(transition)
        if len(self.Weights) < self.batch_size:
            self.Weights.append(1/self.batch_size)
        else : 
            self.Weights.append(max(self.Weights))
        return

    def update_weight(self, chosen_indexes, TD_delta):
        for i in range(len(chosen_indexes)):
            self.Weights[chosen_indexes[i]] = abs(TD_delta[i]) + self.eps

    def compute_probabilities(self):
        probabilities = []
        ws = 0
        for i in range(len(self.Weights)):
            ws += self.Weights[i]**self.alpha

        for i in range(len(self.Weights)):
            probabilities.append((self.Weights[i]**self.alpha) / ws)
        return probabilities

    def sample_batch(self):    
        chosen_indexes = np.arange(len(self.Weights))
        chosen_indexes = np.random.choice(chosen_indexes, self.batch_size, p=self.compute_probabilities())

        batch = []
        for i in chosen_indexes :
            batch.append(self.buf[i])

        return batch, chosen_indexes
