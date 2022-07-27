import os as var_os
import math as var_math
import numpy as var_numpy
import pandas as var_pandas
import pickle as var_pickle
from sklearn import linear_model

def gradient_descent(math, compsci, max_tolerant_cost):
    m_curr = b_curr = 0.00
    cost_curr =  var_numpy.double(0.00)
    cost_prev = cost_diff = None
    num_elements = math.size

    num_iterations  = 500000
    print_diag_rate = 500000
    learning_rate = 2.111/(10**4)

    print("init: element count = {}, iteration count = {}, learning rate = {}, max_tolerant_cost= {}".format(num_elements, num_iterations, learning_rate,max_tolerant_cost))

    for i in range(num_iterations):
        # Prediction based on current coef and intercepts
        predicted_compsci = m_curr * math + b_curr

        # How accurate are we?
        cost_prev = cost_curr
        cost_curr = var_numpy.double((1/num_elements) * sum([value**2 for value in (compsci-predicted_compsci)]))
        if (cost_prev != None): cost_diff =  cost_prev - cost_curr

        # Print results
        if ((0 == i % print_diag_rate) or (i==num_iterations-1)):
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))

        # Are we improving much, if not may be quit?
        if (var_math.isclose(a=cost_prev, b=cost_curr,rel_tol=(10**-20))):
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))
            print("break: reason = cost is close")
            return m_curr, b_curr

        if (cost_curr < max_tolerant_cost): 
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))
            print("break: reason = cost fell below max tolerant cost")
            return m_curr, b_curr
        # if (cost_diff < 0): 
        #     print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
        #     print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))
        #     print("break: reason = cost is increasing")
        #     return m_curr, b_curr

        # Prepare for next iteration
        m_diff = -(2/num_elements) * sum(math*(compsci-predicted_compsci))
        b_diff = -(2/num_elements) * sum(compsci-predicted_compsci)
        m_curr = m_curr - learning_rate * m_diff
        b_curr = b_curr - learning_rate * b_diff

    # All iterations are done, return whatever we have
    return m_curr, b_curr

def sklearn_gradient_descent(math, compsci):
    df = var_pandas.DataFrame({'math' : math, 'compsci' : compsci})
    var_model = linear_model.LinearRegression()
    var_model.fit(df[['math']], df.compsci)
    return var_model

# Main Code
# Caculate Coeffecient (m) and Intercept (b) using Gradient Descent Algorithm

# Data-Set-1
math =   var_numpy.array([92,56,88,70,80,49,65,35,66,67])
compsci = var_numpy.array([98,68,81,80,83,52,66,30,68,73])

# Calculate With Custom Implementation
m_custom,b_custom=gradient_descent(math, compsci, var_numpy.double(25.0))

# Calculate With SciKit Learn Implementation
model_system = sklearn_gradient_descent(math,compsci)
m_system,b_system=model_system.coef_, model_system.intercept_

# Save Model using Pickle
repo_root_path = var_os.path.abspath(var_os.path.dirname(var_os.path.dirname(__file__)))
model_file_path = repo_root_path + "/.outputs/.models/cbex-gd-student-marks-model.bindata"
print ("Model File Path = {}".format(model_file_path))
with open(model_file_path, "wb") as dump_file:
    var_pickle.dump(model_system, dump_file)

# Load Model using Pickle
with open(model_file_path, "rb") as load_file:
    model_loaded = var_pickle.load(load_file)
m_loaded,b_loaded=model_loaded.coef_, model_loaded.intercept_

print ("===== Results =====")
print ("Custom: m = {}, b = {}".format(m_custom,b_custom))
print ("System: m = {}, b = {}".format(m_system,b_system))
print ("Loaded: m = {}, b = {}".format(m_loaded,b_loaded))
