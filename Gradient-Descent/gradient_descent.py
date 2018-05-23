import matplotlib.pyplot as plt


def f(x):

    return x**4 - (2 * x) ** 2 + (4 * x) ** 2 - 10


def g(x):

    return (4 * x) ** 3 - (4 * x) + 4


iteration = 0
alpha = 0.0001
x = 2
precision = 0.00000001

y_axis = []

while x > precision:
    x = x - g(x) * alpha
    print("Iteration: {} | Cost: {}".format(iteration, x))
    iteration += 1
    y_axis.append(x)

x_axis = list(range(0, iteration))

plt.plot(x_axis, y_axis)
plt.show()
