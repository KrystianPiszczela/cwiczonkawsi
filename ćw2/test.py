from ES import ES, init_population

search_for_min = True       # True oznacza szukanie minimum, False - maksimum
iterations_limit = 50        # Limit iteracji
mutation_stddev = 0.1        # Odchylenie standardowe mutacji

combination_list = [(1, 1), (1, 16), (16, 1), (16, 16), (128, 512), (3, 5), (5, 3), (5, 25), (25, 5),
                    (64, 256), (256, 64), (256, 1024), (1024, 256), (1024, 1024)]
number_of_tests = 30

for (N, lambda_) in combination_list:
    l_min_found = 0
    g_min_found = 0
    min_missed = 0
    for i in range(number_of_tests):
        best_values, best_individual = ES(init_population(N))
        if round(best_values, 1) == -2.7:
            g_min_found += 1
        elif round(best_values, 1) == -1.3:
            l_min_found += 1
        else:
            min_missed += 1
    print(f'Dla pary N={N}, lambda={lambda_}: l_min_found={l_min_found}, g_min_found={g_min_found}, min_missed={min_missed}')
