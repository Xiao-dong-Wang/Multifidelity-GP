import autograd.numpy as np
from GP import GP

class NAR_GP:
    def __init__(self, dataset, bfgs_iter=100, debug=True):
        self.low_x = dataset['low_x']
        self.low_y = dataset['low_y']
        self.high_x = dataset['high_x']
        self.high_y = dataset['high_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug

    def train(self):
        dataset = {}
        dataset['train_x'] = self.low_x
        dataset['train_y'] = self.low_y
        model1 = GP(dataset, bfgs_iter=self.bfgs_iter, debug=self.debug)
        model1.train()
        self.model1 = model1
        
        mu, v = self.model1.predict(self.high_x)
        dataset['train_x'] = np.concatenate((self.high_x, mu.reshape(1,-1)))
        dataset['train_y'] = self.high_y
        model2 = GP(dataset, bfgs_iter=self.bfgs_iter, debug=self.debug, k=1)
        model2.train()
        self.model2 = model2
        print('NAR_GP. Finish training process')

    def predict(self, test_x):
        py1, ps21 = self.model1.predict(test_x)
        x = np.concatenate((test_x, py1.reshape(1, -1)))
        py,  ps2 = self.model2.predict(x)
        return py, ps2


    def predict_for_wEI(self, test_x):
        nsamples = 100
        num_test = test_x.shape[1]
        py1, ps21 = self.model1.predict(test_x, is_diag=0)
        Z = np.random.multivariate_normal(py1, ps21, nsamples)
        if self.debug:
            print('Z.shape',Z.shape)
            print('Z[0,:].shape', Z[0,:].shape)
            print('Z[0,:][None,:].shape', Z[0,:][None,:].shape)

        x = np.tile(test_x, nsamples)
        x = np.concatenate((x, Z.reshape(1,-1)))
        py, ps2 = self.model2.predict(x)
        py = py.reshape(-1,num_test)
        ps2 = ps2.reshape(-1,num_test).mean(axis=0) + py.var(axis=0)
        py = py.mean(axis=0)
        return py, ps2
        


    def predict_low(self, test_x):
        return self.model1.predict(test_x)



