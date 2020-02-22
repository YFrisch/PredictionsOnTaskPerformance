import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math


# Functions
def gaussian(x, mu, sigma):
    return (1.0/(sigma*2*np.sqrt(np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)


def uni(x):
    return 10.0/len(x)


def expo(x):
    return 1/np.exp(x)


def weibull(x, alpha, beta):
    return alpha*beta*x**(beta-1)*np.exp(-alpha*x**beta)


def alpha(x, alpha):
    return alpha*np.exp(-7*alpha*x)


# Data
x = np.arange(0, 1, 0.015)
g_x = gaussian(x, mu=0.666, sigma=0.3)
u = uni(x)
u_x = [u for xs in x]
w_x = weibull(x, 1, 1)
a_x = alpha(x, 1)


# Plots
plt.figure()
plt.plot(x, g_x, color='black')
plt.ylim(0, 1)
plt.figure()
plt.plot(x, u_x, color='black')
plt.ylim(0, 1)

plt.figure()
plt.plot(x, a_x, color='black')

plt.show()
