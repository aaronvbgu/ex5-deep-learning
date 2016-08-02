import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RNN:

    # init random network
    def __init__(self, train_size, hidden_layer_size=100, back_propagation=4):
        self.train_size = train_size
        self.hidden_layer_size = hidden_layer_size
        self.back_propagation = back_propagation
        self.U = np.random.uniform(-np.sqrt(1./train_size), np.sqrt(1./train_size), (hidden_layer_size, train_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_layer_size), np.sqrt(1./hidden_layer_size), (train_size, hidden_layer_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_layer_size), np.sqrt(1./hidden_layer_size), (hidden_layer_size, hidden_layer_size))

    # predict words probabilities
    def forward_propagation(self, time_steps):
        T = len(time_steps)
        s = np.zeros((T + 1, self.hidden_layer_size))
        s[-1] = np.zeros(self.hidden_layer_size)
        o = np.zeros((T, self.train_size))
        for time_step in np.arange(T):
            s[time_step] = np.tanh(self.U[:,time_steps[time_step]] + self.W.dot(s[time_step-1]))
            o[time_step] = softmax(self.V.dot(s[time_step]))
        return [o, s]

    # get highest score
    def predict(self, time_steps):
        o, s = self.forward_propagation(time_steps)
        return np.argmax(o, axis=1)