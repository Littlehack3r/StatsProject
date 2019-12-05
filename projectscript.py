# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from matplotlib import pyplot as plt


# %%


# %% [markdown]
# ## Setup: x values
# Define the 'true' function (and test functions).
# Define the x values (xvals)

# %%
# This is the data that we were given. It is inputted as a 2D array
real_data = np.array([[-1,	-4/5,	-3/5,	-2/5,	-1/5,	0,	1/5,	2/5,	3/5,	4/5,	1] , [-8.259416503478342,
             -6.824790633539753,	-2.940304263374559,	-2.0901353112933863,	1.2015719668831903,
               4.838039128178995,	3.5609782071742138,	3.651756599989234,	5.485459444665049,	7.618771800923425,	7.4177051254457895]])

# Print statement isn't necessary I believe, but it may help with debugging
print(real_data)


# %%
xvals = np.linspace(-1,1, 11)
fn1 = lambda x: 3*np.ones(len(xvals))
fn2 = lambda x: -2+3.*x
fn3 = lambda x: 2*x**2
fnTrue =lambda x: 2+10*x-2*x**2


# %%
def makeFakeData(fn):
       return fn(xvals)+np.random.normal(size=len(xvals))


# %%
np.polyfit(xvals,makeFakeData(fn1),2)


# %%
import numpy as np
from math import *
dat_out = []
for indx in np.arange(1000):
    dat_out.append(np.polyfit(xvals,makeFakeData(fn2),1))
dat_out = np.array(dat_out)
print(dat_out)
print(np.mean(dat_out[:,0]))
print(np.mean(dat_out[:,1])) 
print(np.std(dat_out[:,0]))
print(np.cov(dat_out[:,0],dat_out[:,1]))


print("Var C1: " + str(np.var(dat_out[:,0])))
print("Var C2: " + str(np.var(dat_out[:,1])))
print("Cov is: " + str(np.cov(dat_out[:,0],dat_out[:,1])))

# This is where the calculations for the variance of the A7 given data start:
sum_of_cov = np.cov(dat_out[:,0],dat_out[:,1])
print(sum_of_cov)


# Not sure if these print statements are needed for project, added them just in case
#print("Cov A7 Data C0: " + str(np.cov(real_data[:,0])))
#print("Cov A7 Data C1: " + str(np.cov(real_data[:,1])))

# This is the calculated variance
f_variance = np.var(sum_of_cov)
print("F Variance: ")
print(f_variance)

plt.scatter(dat_out[:,0],dat_out[:,1])
plt.show()


numbers =[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
avg = np.average(numbers)
temp = 1.645 * f_variance
lamb = temp/sqrt(11)
upBound = avg + lamb
lowBound = avg - lamb
avgstr = str(avg)
strubound = str(upBound)
strlbound = str(lowBound)
print("Average: " + avgstr)
print("Up Bound: " + strubound)
print("Low Bound: " + strlbound)


# %%
#def mygauss(x, mu, sigma):
#    np.exp(- np.power(x-mu,2)/(2.*sigma**2) ) /np.sqrt(2*np.pi)
xvalsTest = np.linspace(-20,20,500)
#yvalsTest =  mygauss(xvalsTest,0,1)
yvalsTest = np.exp( - np.power(xvalsTest-0,2)/(2.*1**2) )/ np.sqrt(2*np.pi)
plt.plot(xvalsTest,yvalsTest)
plt.savefig("my_fig.png")
plt.xlabel("Hello there")
plt.ylabel(r'${\cal L}_m^\gamma$')
plt.show()


# %%


# %% [markdown]
# def mygauss(x, mu, sigma):
#     np.exp(- (x-mu)**2/(2.*sigma**2) ) /np.sqrt(2*np.pi)
# mygauss(np.linspace(-20,20),0,1)
# %% [markdown]
# ## Create and plot data

# %%
yExamples =makeFakeData(fn3)


# %%
plt.plot(xvals,yExamples,'o');
plt.show()

# %% [markdown]
# ## Fit quadratic, plot quadratic and fit
# Words for fun and profit

# %%
z = np.polyfit(xvals,yExamples,2)
z
fn = np.poly1d(z)


# %%
zvalsMany = np.array(
    [np.polyfit(xvals,makeFakeData(fn1),2) for k in np.arange(30)]
)


# %%
c0hatV1 = [c0 for c0,c1,c2 in zvalsMany]
c0hatV2 = zvalsMany[:,0]
c1hatV2 = zvalsMany[:,1]
np.mean(c0hatV2)
indx = np.arange(len(c0hatV1))
result =0
for i in indx:
    result += (c0hatV2[indx] - np.mean(c0hatV2))*(c1hatV2[indx] - np.mean(c1hatV2))
result = result/(1.0*len(c0hatV1))


# %%
np.std(c0hatV2)


# %%
xvals_show = np.linspace(-1,1,200);
plt.plot(xvals,yExamples,'o')
plt.plot(xvals_show,fn(xvals_show))
plt.show()


# %%
def get_random_coefs(deg):
    yvals = makeFakeData(fnTrue)
    return np.polyfit(xvals,yvals, deg) 


# %%
get_random_coefs(2)


# %%



