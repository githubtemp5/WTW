
# coding: utf-8

# In[4]:


import pandas as pd

df = pd.read_csv('C:/Users/ShresthaAl/Downloads/PMI_dataset_v2.txt',
                 header=None,
                 sep='\s+')

df.columns = ['EXIST', 'NEW', 'LAPSED']
df.head()


# In[3]:


# Visualizing the important characteristics of a dataset
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


cols = ['EXIST', 'NEW', 'LAPSED']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()


# In[4]:


import numpy as np


cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()


# In[5]:


# Solving regression for regression parameters with gradient descent
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# In[6]:


X = df[['NEW']].values
y = df['EXIST'].values


# In[7]:


from sklearn.preprocessing import StandardScaler


sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# In[8]:


lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# In[9]:


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.tight_layout()
#plt.savefig('images/10_05.png', dpi=300)
plt.show()


# In[10]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return


# In[11]:


lin_regplot(X_std, y_std, lr)
plt.xlabel('New Business')
plt.ylabel('Existing Book of Business')

#plt.savefig('images/10_06.png', dpi=300)
plt.show()


# In[12]:


print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# In[13]:


New_Business_std = sc_x.transform(np.array([[5.0]]))
Existing_std = lr.predict(New_Business_std)
print("Existing Book of Business: %.3f" % sc_y.inverse_transform(Existing_std))


# In[14]:


# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression


# In[15]:


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# In[16]:


lin_regplot(X, y, slr)
plt.xlabel('New Business')
plt.ylabel('Existing Book of Business')

#plt.savefig('images/10_07.png', dpi=300)
plt.show()


# In[17]:


# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])


# In[18]:


# print('Slope: %.3f' % ransac.estimator_.coef_[0])
# print('Intercept: %.3f' % ransac.estimator_.intercept_)


# In[19]:


import numpy as np
import scipy as sp

ary = np.array(range(100000))


# In[39]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE Data: %.3f' % (mean_squared_error(y, y_pred)))
print('R^2 Data: %.3f' % (r2_score(y, y_pred)))


# In[40]:


# Using regularized methods for regression
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
y_pred = lasso.predict(X)
print(lasso.coef_)


# In[41]:


from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)


# In[42]:


# fit linear features
lr.fit(X, y)
X_fit = np.arange(4000, 10000, -10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# fit quadratic features
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plot results
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/10_10.png', dpi=300)
plt.show()


# In[43]:


y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)


# In[44]:


print('Training MSE linear: %.3f, quadratic: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))


# In[45]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)


# In[46]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)


# In[49]:


from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)


# In[51]:


# Dealing with nonlinear relationships using random forests
# Decision tree regression
from sklearn.tree import DecisionTreeRegressor

X = df[['NEW']].values
y = df['EXIST'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('New Business')
plt.ylabel('Existing Book of Business')
#plt.savefig('images/10_13.png', dpi=300)
plt.show()


# In[1]:


# Random forest regression
X = df.iloc[:, :-1].values
y = df['EXIST'].values

#X_train, X_test, y_train, y_test = train_test_split(
 #   X, y, test_size=0.4, random_state=1)

