# Vanilla Gradient Descent for Linear Regression
# ----------------------------------------------

import numpy as np


def gradient_descent(b0, b1, y, x, alpha):

    # Inputs to gradient descent function:
    # b0: Intercept of the linear function
    # b1: Slope
    # y: Vector with response values
    # x: Vector with predictor values
    # alpha: Learning rate

    # Update coefficients for each value in the data
    def compute_gradients(b0, b1, y, x, alpha):

        # Number of observations
        n = float(len(y))

        # Set gradient vectors to zero
        b0_gradient = 0
        b1_gradient = 0

        # b0_gradient: Partial derivative in respect to b0
        # b1_gradient: Partial derivative in respect to b1

        b0_gradient = b0_gradient - (2 / n) * (y - ((b1 * x) + b0))
        b1_gradient = b1_gradient - (2 / n) * x * (y - ((b1 * x) + b0))

        # Update coefficients at each iteration by subtracting the
        # product of the learning rate and the gradient from the
        # current coefficient.
        b0 = b0 - (alpha * b0_gradient)
        b1 = b1 - (alpha * b1_gradient)

        return [b0, b1]

    def error(b0, b1, y, x):
        return (np.sum(y - (b1 * x + b0)) ** 2) / float(len(y))

    # Main function
    # Compute error for the input data
    mse = error(b0, b1, y, x)
    # Initialize error logger with the last error_sum
    error_logger = [mse]
    # Initialize empty coefficient logging lists
    b0_logger = []
    b1_logger = []
    # Count iterations
    iteration = 1
    # For each iteration ensure that error is decreasing
    while mse == min(error_logger):
        # Compute new coefficients
        coefficients = compute_gradients(b0, b1, y, x, alpha)
        # Update coefficients
        b0 = coefficients[0]
        b1 = coefficients[1]
        # Log coefficients
        b0_logger.append(b0)
        b1_logger.append(b1)
        # Compute error
        mse = error(b0, b1, y, x)
        print(mse)
        # Log error
        error_logger.append(mse)
        # Increment iteration counter
        iteration += 1

    return [b0, b1, b0_logger, b1_logger, mse, error_logger, iteration]


x = np.random.rand(500, 1)
y = np.random.rand(500, 1)

b0 = np.array(4)
b1 = np.array(3)

alpha = np.array(0.000001)

output = gradient_descent(b0, b1, x, y, alpha)

