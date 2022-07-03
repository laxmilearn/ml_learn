import numpy as var_numpy
import pandas as var_pandas

def gradient_descent(area, price, max_tolerant_cost):
    m_curr = b_curr = 0.00
    cost_curr = cost_prev = cost_diff = var_numpy.double(0.00)
    num_elements = area.size

    num_iterations  = 100000000
    print_diag_rate = 1000000
    learning_rate = 13.75/(10**8)

    print("init: element count = {}, iteration count = {}, learning rate = {}".format(num_elements, num_iterations, learning_rate))

    for i in range(num_iterations):
        # Prediction based on current coef and intercepts
        predicted_price = m_curr * area + b_curr

        # How accurate are we?
        cost_prev = cost_curr
        cost_curr = var_numpy.double((1/num_elements) * sum([value**2 for value in (price-predicted_price)]))
        if cost_prev == var_numpy.double(0.00): cost_prev = cost_curr
        cost_diff =  cost_prev - cost_curr

        # Print results
        if ((0 == i % print_diag_rate) or (i==num_iterations-1)):
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))

        # Are we improving much, if not may be quit?
        if (cost_curr < max_tolerant_cost): 
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))
            print("break: reason = cost fell below max tolerant cost")
            break
        if (cost_diff < 0): 
            print("iteration = {}: m = {}, b = {}".format(i, cost_curr, m_curr, b_curr))
            print("cost: prev = {}, curr = {}, diff = {}".format(cost_prev, cost_curr, cost_diff))
            print("break: reason = cost is increasing")
            break

        # Prepare for next iteration
        m_diff = -(2/num_elements) * sum(area*(price-predicted_price))
        b_diff = -(2/num_elements) * sum(price-predicted_price)
        m_curr = m_curr - learning_rate * m_diff
        b_curr = b_curr - learning_rate * b_diff

# Main Code
# train_df = var_pandas.read_csv("../data/cbex-lr-home-prices-train.csv")
# area = var_numpy.array(train_df.area)
# price = var_numpy.array(train_df.price)

# Data-Set-1
# area = var_numpy.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000])
# price = var_numpy.array([80000,135000,240000,325000,450000,560000,800000,945000,1100000,1250000,1400000])
# Tuned Values for above tough data-set-1
# learning_rate = 6.07733/(10**8)

# Data-Set-2
#area = var_numpy.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
#price = var_numpy.array([80000,135000,240000,325000,450000,560000,800000,945000,1100000])
# Tuned Values for this tough data-set-2
# learning_rate = 9.374999/(10**8)

# Data-Set-3
area = var_numpy.array([   1000,   1500,   2000,   2500,   3000,   3500,   4000])
price = var_numpy.array([105000, 155000, 205000, 255000, 305000, 355000, 405000])
# Tuned Values for this simple data-set-3
# Should ideally reach m=100, b=500
# learning_rate = 13.75/(10**8)

gradient_descent(area, price, var_numpy.double(1000.0))

