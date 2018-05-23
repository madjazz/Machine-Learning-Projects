# Stochastic Gradient Descent for Linear Regression
# -------------------------------------------------

import random

def gradient_descent(b0, b1, y, x, alpha):

    # Inputs to gradient descent function:
    # b0: Intercept of the linear function
    # b1: Slope
    # y: Vector with response values
    # x: Vector with predictor values
    # alpha: Learning rate

    # Update coefficients for each value in the data


    def error(b0, b1, y, x):
        n = len(y)
        total_error = y[0] - (b1 * x[0] + b0) ** 2
        for i in range(1, n):
            total_error = total_error + (y[i] - (b1 * x[i] + b0)) ** 2
        return total_error / len(y)

    def step_gradient(b0, b1, y, x, alpha):
        # Set gradients to zero
        b0_gradient = 0
        b1_gradient = 0

        # Update gradients for N iterations
        n = len(y)

        # Randomly shuffle data

        random.shuffle(y)
        random.shuffle(x)

        for item_x, item_y in zip(x, y):
            # b0_gradient: Partial derivative in respect to b0
            # b1_gradient: Partial derivative in respect to b1
            b0_gradient = b0_gradient - (2 / n) * (item_y - ((b1 * item_x) + b0))
            b1_gradient = b1_gradient - (2 / n) * item_x * (item_y - ((b1 * item_x) + b0))

            # Update coefficients at each iteration by subtracting the
            # product of the learning rate and the gradient from the
            # current coefficient.
            b0 = b0 - (alpha * b0_gradient)
            b1 = b1 - (alpha * b1_gradient)

        return [b0, b1]


    # Main function
    # Compute error for the input data
    mse = error(b0, b1, y, x)

    # Initialize error logger with the last error_sum
    error_logger = [mse]

    # Initialize empty coefficient logging vectors
    b0_logger = []
    b1_logger = []

    # Count iterations
    iteration = 1

    # For each iteration ensure that error is decreasing
    while mse == min(error_logger):
        # Compute new coefficients
        coefficients = step_gradient(b0, b1, y, x, alpha)

        # Update coefficients
        b0 = coefficients[0]
        b1 = coefficients[1]

        # Log coefficients
        b0_logger.append(b0)
        b1_logger.append(b1)

        # Compute error
        mse = error(b0, b1, y, x)

        # Log error
        error_logger.append(mse)

        # Increment iteration counter
        iteration += 1

    return {"b0": b0, "b1": b1, "b0_logger": b0_logger, "b1_logger": b1_logger,
            "mse": mse, "error_logger": error_logger, "iteration": iteration}


x = random.sample(range(500), 200)
y = random.sample(range(500), 200)

b0 = 1
b1 = 0

alpha = 0.000001

gradient_descent(b0, b1, y, x, alpha)
