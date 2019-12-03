
import numpy as np
from matplotlib import pyplot as plt

# Set X values


xvals = np.linspace(-1,1, 11)
fn1 = lambda x: 3*np.ones(len(xvals))
fn2 = lambda x: -2+3.*x
fn3 = lambda x: 2*x**2
fnTrue = lambda x: 2+10*x-2*x**2

def makeFakeData(fn):
       return fn(xvals)+np.random.normal(size=len(xvals))
