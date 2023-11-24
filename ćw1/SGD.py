from math import e
from random import uniform

search_for_min = False  # True oznacza szukanie minimum, False - maksimum
learning_rate = 0.01
iterations_limit = 100000


def function_value(x, y):
    # return (10*x*y)/(e^(x^2+x/2+y^2))
    return (10*x*y)/(e**(x**2+x/2+y**2))


def SGD(starting_point, learning_rate, iterations_limit):
    x, y = starting_point
    for i in range(iterations_limit):
        partial_derivative_x = 10*y*(1/(e**(x**2+x/2+y**2))+(-2*x-0.5)*x*1/(e**(x**2+x/2+y**2)))
        partial_derivative_y = 10*x*(1-2*y**2)*(1/(e**(x**2+x/2+y**2)))
        if search_for_min is True:
            x = x - learning_rate * partial_derivative_x
            y = y - learning_rate * partial_derivative_y
        else:
            x = x + learning_rate * partial_derivative_x
            y = y + learning_rate * partial_derivative_y
    return x, y


def print_results(x, y, starting_point):
    print(f'Punkt startowy to: {starting_point[0]}, {starting_point[1]}')
    print(f'Wartość x, dla której f(x, y) jest minimalne: {x}')
    print(f'Wartość y, dla której f(x, y) jest minimalne: {y}')
    print(f'Minimum funkcji: {function_value(x, y)}')


def main():
    starting_point = 10, 10
    # starting_point = uniform(-2, 2), uniform(-2, 2)
    x, y = SGD(starting_point, learning_rate, iterations_limit)
    print_results(x, y, starting_point)


main()
