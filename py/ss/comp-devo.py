from scipy.special import expit
import scipy.optimize
from scipy.optimize import minimize, differential_evolution
import numpy as np
from math import sin,cos
#xarr=var('xarr')
x=np.linspace(0,2,11)
#x=np.array([1.0])
hvar=5
numeqs=4
omega=1.0
theta=1.0
bounds=np.zeros([3*4*5,2])
for i in range(3*4*5):
    bounds[i,0]=-5
    bounds[i,1]=5
partot=np.array(np.zeros(3*hvar*numeqs))
x0=[1.0,0.0,0.0,0.0]
#par = par.reshape(3,hvar)
print partot
print x
one=np.ones(hvar)

def sig(x,par):
    ans=[]
    par1 = par.reshape(3,hvar)
    #print "test", par[2]
    for i in x:
        ans.append(expit(i*par1[1,:]+par1[2,:]))
        #ans.append(np.tanh(i*par1[1,:]+par1[2,:]))
    return ans
def N(x,par):
    par1=par.reshape(3,hvar)
    ans=np.inner(par1[0,:],sig(x,par))
    return ans
def y(x,par,xini):
    return xini+x*N(x,par)
def dNdx(x,par):
    par1=par.reshape(3,hvar)
    ans=np.zeros(len(x))
    #print len(x)
    for j in range(len(x)):
        for i in range(hvar):
            ans[j]=ans[j]+(par1[0,i])*(sig(x,par)[j][i])*((one-sig(x,par))[j][i])*par1[1,i]
    return(ans)
def dydx(x,par):
    return N(x,par)+x*dNdx(x,par)
def yp(partot):
    partot1=partot.reshape((numeqs,3,hvar))
    cost=0.0
    cost=cost+np.sum(0.5*(dydx(x,partot1[0,:,:])-2*omega*sin(2*theta)*y(x,partot1[3,:,:],x0[3]))**2)
    cost=cost+np.sum(0.5*(dydx(x,partot1[1,:,:])+2*omega*sin(2*theta)*y(x,partot1[3,:,:],x0[3]))**2)
    cost=cost+np.sum(0.5*(dydx(x,partot1[2,:,:])-2*omega*cos(2*theta)*y(x,partot1[3,:,:],x0[3]))**2)
    cost=cost+np.sum(0.5*(dydx(x,partot1[3,:,:])+2*omega*cos(2*theta)*y(x,partot1[2,:,:],x0[2])+omega*sin(2*theta)*y(x,partot1[0,:,:],x0[0])-omega*sin(2*theta)*y(x,partot1[1,:,:],x0[1]))**2)
    cost = cost#+np.sum((y(x,partot1[0,:,:],x0[0])+y(x,partot1[1,:,:],x0[1])-1.0)**2)


    return cost
#def ypprime(par): 
#vout=minimize(yp,par,method='COBYLA',options={"maxfev": 10000})
#vout=minimize(yp,partot,method='SLSQP',options={"maxiter": 1000})
#vout=minimize(yp,partot,method='Nelder-Mead',tol=1e-7,options={"ftol":1e-3, "maxfev": 1000000,"maxiter":1000000})
vout=differential_evolution(yp,bounds,strategy='best1bin',tol=0.00001,maxiter=1000000,polish=True)
print vout


f = open("vout-devo-file.txt", 'w')
f.write(vout)
f.close()

np.savetxt('vout-devo.txt', vout, delimiter=',')

