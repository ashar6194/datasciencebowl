from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np


def regression_coef(A, train_y):
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), train_y)


def ridge_regression_coef(A, train_y, alpha):
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), A.T), train_y)


boston = load_boston()
X = boston.data
y = boston.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.208, random_state=42)

# Find mean and std dev for the train set
mean_arr = np.mean(train_X, axis=0)
std_arr = np.std(train_X, axis=0)

# Normalize train_set
norm_train_X = (train_X - mean_arr)/std_arr
A = np.concatenate([np.ones((norm_train_X .shape[0], 1)), norm_train_X], axis=1)

theta = regression_coef(A, train_y)

# Normalize test set
norm_test_X = (test_X - mean_arr)/ std_arr
test_X_homog = np.concatenate([np.ones((norm_test_X.shape[0], 1)), norm_test_X ], axis=1)

# Find MSE - Regression
err = 0
for a, b in zip(test_X_homog, test_y):
    y_pred = np.dot(theta, a)
    err += (b - y_pred)**2
mse_regression = err/(test_X.shape[0])

print '\n\nSIMPLE REGRESSION: MSE for is ' + str(mse_regression) + '\n\n'

# Implementation of Ridge Regression
X_mini_train, X_holdout, y_mini_train, y_holdout = train_test_split(A, train_y, test_size=0.25, random_state=42)

alpha_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]

for alpha in alpha_list:
    theta = ridge_regression_coef(X_mini_train, y_mini_train, alpha)
    # Find MSE - Ridge Regression
    err = 0

    for a, b in zip(X_holdout, y_holdout):
        y_pred = np.dot(theta, a)
        err += (b - y_pred) ** 2
    mse_ridge = err / (X_holdout.shape[0])
    print 'Value of lambda - ' + str(alpha) + ', MSE = ' + str(mse_ridge)

alpha_final = 1
theta = ridge_regression_coef(A, train_y, alpha_final)

# Find MSE (Test Pipeline) - Ridge Regression
err = 0
for a, b in zip(test_X_homog, test_y):
    y_pred = np.dot(theta, a)
    err += (b - y_pred) ** 2
mse_ridge = err / (test_X.shape[0])

print '\n\nRIDGE REGRESSION: Value for lambda = ' + str(alpha_final) + ', MSE = ' + str(mse_ridge) + '\n\n'


# Lasso Regression Loop
# print '\n\n Carrying out Lasso regression now\n'

X_mini_train, X_holdout, y_mini_train, y_holdout = train_test_split(norm_train_X, train_y, test_size=0.25, random_state=42)

alpha_list_lasso = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

for alpha in alpha_list_lasso:
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(X_mini_train, y_mini_train)

    y_pred = reg.predict(X_holdout)
    mse_lasso = mean_squared_error(y_holdout, y_pred)
    print 'Value of lambda - ' + str(alpha) + ', MSE = ' + str(mse_lasso)


alpha_final_lasso = 0.1
reg = linear_model.Lasso(alpha=alpha_final_lasso)
reg.fit(norm_train_X, train_y)

y_pred = reg.predict(norm_test_X)
mse_lasso = mean_squared_error(test_y, y_pred)
print '\n\nLASSO REGRESSION: Value of lambda - ' + str(alpha_final_lasso) + ', MSE = ' + str(mse_lasso)

theta_final = reg.coef_
print 'Number of non-zero coefficients is = ' + str(np.count_nonzero(theta_final))