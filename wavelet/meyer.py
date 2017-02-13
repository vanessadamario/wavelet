import numpy as np


def aux(x):
    pol = np.poly1d([-20, 70, -84, 35, 0, 0, 0, 0])
    y = pol(x) * (x>=0) * (x<=1) #+ (x>1);
    return y

def mother(x):
    x = np.abs(x)
    int1 = (x > np.pi/4) & (x <= np.pi/2);
    int2 = (x > np.pi/2) & (x <= np.pi);
    y =  int1 * np.sin(np.pi/2*aux(4*x/np.pi-1)) #* np.exp(1j*4/3*x);
    y = y + int2 * np.cos(np.pi/2*aux(2*x/np.pi-1)) #* np.exp(1j*4/3*x);
    return y

def scaling(x):
    x = np.abs(x)
    int1 = x < np.pi/4
    int2 = (x > np.pi/4) & (x < np.pi/2)
    y = int1 * np.ones(len(x)) + int2 * np.cos(np.pi/2 * aux(4*x/np.pi-1))
    return y
