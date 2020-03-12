import autograd.numpy as np
import sys
import toml
from util import *
from NAR_GP import NAR_GP
from get_dataset import *
import multiprocessing
import pickle
import matplotlib.pyplot as plt

#np.random.seed(34)

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
bfgs_iter = conf['bfgs_iter']

#### MFGP
dataset = init_dataset(funct, num, bounds)
low_x = dataset['low_x']
low_y = dataset['low_y']
high_x = dataset['high_x']
high_y = dataset['high_y']
model = NAR_GP(dataset, bfgs_iter[0], debug=True)
model.train()



# Test data
nn = 200
X_star = np.linspace(-0.5, 0.5, nn)[None,:]
y_star_high = funct[1](X_star,bounds)
y_star_low = funct[0](X_star,bounds)
X_star_real = X_star * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
y_pred, y_var = model.predict(X_star)

low_x_real = low_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
high_x_real = high_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2


plt.figure()
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star_high.flatten(), 'b-', label = "high fidelity", linewidth=2)
plt.plot(X_star_real.flatten(), y_star_low.flatten(), 'g-', label = "low fidelity", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred.flatten(), 'r--', label = "Prediction", linewidth=2)
lower = y_pred - 2.0*np.sqrt(y_var)
upper = y_pred + 2.0*np.sqrt(y_var)
plt.fill_between(X_star_real.flatten(), lower.flatten(), upper.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(low_x_real, low_y, 'go')
plt.plot(high_x_real, high_y, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()






