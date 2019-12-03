
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


def outputData():
       dat_out = []
       for indx in np.arange(500):
       dat_out.append(np.polyfit(xvals,makeFakeData(fn3),2))
       dat_out = np.array(dat_out)
       print dat_out
       print np.mean(dat_out[:,0])
       print np.mean(dat_out[:,1])
       print np.std(dat_out[:,0])
       print np.cov(dat_out[:,0],dat_out[:,1])

plt.scatter(dat_out[:,0],dat_out[:,1])
def variance():
       print np.mean(dat_out[:,0])
       print np.mean(dat_out[:,1])
if __name__ == "__main__":
    np.polyfit(xvals,makeFakeData(fn1),2)
    outputData()
    