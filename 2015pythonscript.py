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
dat_out = []
for indx in np.arange(500):
    dat_out.append(np.polyfit(xvals,makeFakeData(fn2),1))
dat_out = np.array(dat_out)
print(dat_out)
print(np.mean(dat_out[:,0]))
print(np.mean(dat_out[:,1])) 
print(np.std(dat_out[:,0]))
print(np.cov(dat_out[:,0],dat_out[:,1]))

plt.scatter(dat_out[:,0],dat_out[:,1])


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


