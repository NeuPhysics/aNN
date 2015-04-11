import numpy as np
from scipy.optimize import minimize
from scipy.optimize import rosen, differential_evolution
from scipy.special import expit
import matplotlib.pyplot as plt
import scipy

from matplotlib.lines import Line2D

import timeit

import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls


array12 = np.asarray(np.split(np.random.rand(1,60)[0],12))


def act(x):
    return expit(x)


# In[4]:

# Density matrix in the forms that I wrote down on my Neutrino Physics notebook
# x is a real array of 12 arrays.

init = np.array([1.0,0.0,0.0,0.0])

def rho(x,ti,initialCondition):

    elem = np.ones(4)

    for i in np.linspace(0,3,4):
        elem[i] = np.sum(ti*x[i*3]*act(ti*x[i*3+1] + x[i*3+2]) )

    return init + elem


# In[5]:

rho(array12,0,init)


# In[6]:

# Hamiltonian of the problem, in terms of four real components

hamil = np.array( [  np.cos(2.0),np.sin(2.0) , np.sin(2.0),np.cos(2.0) ] )
#hamil = 1.0/2*np.array( [  -np.cos(2.0),np.sin(2.0) , np.sin(2.0),np.cos(2.0) ] )
print hamil


# In[8]:

# Cost function for each time step

def rhop(x,ti,initialCondition):

    rhoprime = np.zeros(4)

    for i in np.linspace(0,3,4):
        rhoprime[i] = np.sum(x[i*3] * (act(ti*x[i*3+1] + x[i*3+2]) ) ) +  np.sum( ti*x[i*3]* (act(ti*x[i*3+1] + x[i*3+2]) ) * (1.0 - (act(ti*x[i*3+1] + x[i*3+2])  ) )* x[i*3+1]  )

    return rhoprime



## This is the regularization

regularization = 0.0001

def costi(x,ti,initialCondition):

    rhoi = rho(x,ti,initialCondition)
    rhopi = rhop(x,ti,initialCondition)

    costTemp = np.zeros(4)

    costTemp[0] = ( rhopi[0] - 2.0*rhoi[3]*hamil[1] )**2
    costTemp[1] = ( rhopi[1] + 2.0*rhoi[3]*hamil[1] )**2
    costTemp[2] = ( rhopi[2] - 2.0*rhoi[3]*hamil[0] )**2
    costTemp[3] = ( rhopi[3] + 2.0*rhoi[2]*hamil[0] - hamil[1] * (rhoi[1] - rhoi[0] ) )**2

    return np.sum(costTemp)# + 2.0*(rhoi[0]+rhoi[1]-1.0)**2


#    return np.sum(costTemp) + regularization*np.sum(x**2)


# In[9]:

costi(array12,0,init)


# In[10]:

def cost(x,t,initialCondition):

    costTotal = map(lambda t: costi(x,t,initialCondition),t)

    return 0.5*np.sum(costTotal)


# In[11]:

cost(array12,np.array([0,1,2]),init)
#cost(xresult,np.array([0,4,11]),init)


# <a id='numpyopt'></a>
# -----
# ## NUMPY OPT

# In[12]:

# with ramdom initial guess. TO make it more viable, I used (-5,5)

initGuess = np.asarray(np.split( 5.0*(np.random.rand(1,60)[0] - 0.5),12))
#initGuess = np.split(np.zeros(60),12)
endpoint = 4
tlin = np.linspace(0,endpoint,11)

costF = lambda x: cost(x,tlin,init)

start = timeit.default_timer()
costvFResult = minimize(costF,initGuess,method="SLSQP",tol=1e-20)
stop = timeit.default_timer()

print stop - start

print costvFResult


# - **Should think about the eps(stepsize)**

# In[ ]:

xmid = costvFResult.get("x")

start = timeit.default_timer()
#costvFResult = minimize(costF,xmid,method="SLSQP",tol=1e-30,options={"ftol":1e-30,"maxiter":100000})
costvFResult = minimize(costF,xmid,method='Nelder-Mead',tol=1e-5,options={"ftol":1e-3, "maxfev": 1000000,"maxiter":1000000})
stop = timeit.default_timer()

print stop - start
timespent = stop - start

print costvFResult


# In[ ]:

xresult = costvFResult.get("x")

print xresult

np.savetxt('costvFResult.txt',costvFResult,delimiter=',')

np.savetxt('xresult.txt', xresult, delimiter = ',')

np.savetxt('timespent.txt', timespent, delimiter = ',')
