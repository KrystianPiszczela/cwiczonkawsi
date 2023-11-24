import time
import main
import random
import math
import matplotlib.pyplot as plt

random.seed(318397 % 42)

process_time = []
first_agent_points = []
second_agent_points = []
win = 0
draw = 0
lose = 0

for i in range(1000):
    start_time = time.time()

    vector = [random.randint(-10, 10) for _ in range(15)]
    first_agent, second_agent = main.MinMaxAgent(50), main.NinjaAgent()
    if i % 2 == 0:
        main.run_game(vector, first_agent, second_agent)
    else:
        main.run_game(vector, second_agent, first_agent)

    end_time = time.time()
    process_time.append(end_time - start_time)
    first_agent_points.append(sum(first_agent.numbers))
    second_agent_points.append(sum(second_agent.numbers))
    if sum(first_agent.numbers) > sum(second_agent.numbers):
        win += 1
    elif sum(first_agent.numbers) == sum(second_agent.numbers):
        draw += 1
    else:
        lose += 1

mean_time = sum(process_time)/len(process_time)
mean_f_a_p = sum(first_agent_points)/len(first_agent_points)
mean_s_a_p = sum(second_agent_points)/len(second_agent_points)

stddev_f_a = math.sqrt(sum([diff ** 2 for diff in [x - mean_f_a_p for x in vector]]) / len(vector))
stddev_s_a = math.sqrt(sum([diff ** 2 for diff in [x - mean_s_a_p for x in vector]]) / len(vector))


def plot_histogram(vector, bins=30):
    plt.hist(vector, bins=bins, color='blue', edgecolor='black')
    plt.title('MinMaxAgent (depth=10) vs MinMaxAgent (50)')
    plt.xlabel('Suma punktów')
    plt.ylabel('Liczba wystąpień')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


print(f"Sredni czas gry: {mean_time}\n"
      f"Srednia liczba punktów pierwszego agenta: {mean_f_a_p}\n"
      f'Srednia liczba punktów drugiego agenta: {mean_s_a_p}\n'
      f'Odchylenie standardowe pierwszego agenta: {stddev_f_a}\n'
      f'Odchylenie standardowe drugiego agenta: {stddev_s_a}')
print(f"Wygrane: {win}, remisy: {draw}, przegrane {lose}")
plot_histogram(first_agent_points)
