import numpy as np
from math import e

search_for_min = True        # True oznacza szukanie minimum, False - maksimum
N = 1024                     # Rozmiar populacji początkowej
iterations_limit = 50        # Limit iteracji
mutation_stddev = 0.1        # Odchylenie standardowe mutacji
lambda_ = 1023              # Rozmiar populacji potomków


def function_value(x, y):
    # return (10*x*y)/(e^(x^2+x/2+y^2))
    return (10*x*y)/(e**(x**2+x/2+y**2))


def init_population(N):
    population = np.tile(np.array([0, 0]), (N, 1))
    #population = np.random.uniform(-10, 10, size=(N, 2))
    return population


def ES(population):
    for iteration in range(iterations_limit):
        calculated_values = function_value(population[:, 0], population[:, 1])

        selected_indices = np.argsort(calculated_values)[:N] if search_for_min else np.argsort(calculated_values)[-N:]
        selected_population = population[selected_indices]

        mutation = np.random.normal(0, mutation_stddev, (lambda_, 2))
        mutated_population = selected_population[:, np.newaxis, :] + mutation

        crossover_indices = np.random.randint(0, N, size=(lambda_, 2))
        a = np.random.rand(lambda_, 1)
        crossover_population = (
            a * selected_population[crossover_indices[:, 0]]
            + (1 - a) * selected_population[crossover_indices[:, 1]]
        )

        new_population = np.vstack((selected_population, mutated_population.reshape(-1, 2), crossover_population))

        values = function_value(new_population[:, 0], new_population[:, 1])
        best_indices = np.argsort(values)[:N] if search_for_min else np.argsort(values)[-N:]
        population = new_population[best_indices]

        best_values = values[best_indices[0]]
        best_individual = population[0]
        print_result(iteration, best_values, best_individual)

    return best_values, best_individual


def print_result(iteration, best_values, best_individual):
    print(f"Iteracja nr {iteration + 1}: Najlepsze dopasowanie = {best_values}, najlepszy osobnik = {best_individual}")


def print_summary(best_values, best_individual):
    print(f"Znalezione minimum: {best_values} w punkcie {best_individual}")


def main():
    best_values, best_individual = ES(init_population(N))
    print_summary(best_values, best_individual)


main()
