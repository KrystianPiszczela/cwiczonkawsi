import random

# random.seed(37)  # TODO: For final results set seed as your student's id modulo 42


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__(OOOO000O000O00000):
        OOOO000O000O00000.numbers = []

    def act(O000000O000OO0O0O, O0OO0O0O0O0OO0O00: list):
        if len(O0OO0O0O0O0OO0O00) % 2 == 0:
            O00O0O0000000OO0O = sum(O0OO0O0O0O0OO0O00[::2])
            O0O00O0OO00O0O0O0 = sum(O0OO0O0O0O0OO0O00) - O00O0O0000000OO0O
            if O00O0O0000000OO0O >= O0O00O0OO00O0O0O0:
                O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[0])
                return O0OO0O0O0O0OO0O00[1:]  # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[-1])
            return O0OO0O0O0O0OO0O00[:-1]
        else:
            O00O0O0000000OO0O = max(sum(O0OO0O0O0O0OO0O00[1::2]), sum(O0OO0O0O0O0OO0O00[2::2]))
            O0O00O0OO00O0O0O0 = max(sum(O0OO0O0O0O0OO0O00[:-1:2]), sum(O0OO0O0O0O0OO0O00[:-2:2]))
            if O00O0O0000000OO0O >= O0O00O0OO00O0O0O0:
                O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[-1])
                return O0OO0O0O0O0OO0O00[:-1]
            O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[0])
            return O0OO0O0O0O0OO0O00[1:]


class MinMaxAgent:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.numbers = []

    def act(self, vector: list):
        if len(vector) == 1:
            vector = self.choose_first(vector)
            return vector
        else:
            value_if_first = self.minmax(vector[1:], False, self.max_depth - 1)
            value_if_first += vector[0]
            value_if_last = self.minmax(vector[:-1], False, self.max_depth - 1)
            value_if_last += vector[-1]
            vector = self.choose_first(vector) if value_if_first > value_if_last else self.choose_last(vector)
        return vector

    def choose_first(self, vector):
        self.numbers.append(vector[0])
        return vector[1:]

    def choose_last(self, vector):
        self.numbers.append(vector[-1])
        return vector[:-1]

    def minmax(self, vector, max_player, depth):
        if depth == 0 or len(vector) == 1:
            return max(vector[0], vector[-1]) if max_player else -max(vector[0], vector[-1])
        else:
            value_if_first = self.minmax(vector[1:], not max_player, depth - 1)
            value_if_last = self.minmax(vector[:-1], not max_player, depth - 1)
            if max_player:
                value_if_first += vector[0]
                value_if_last += vector[-1]
                chosen_value = max(value_if_first, value_if_last)
            else:
                value_if_first -= vector[0]
                value_if_last -= vector[-1]
                chosen_value = min(value_if_first, value_if_last)
            return chosen_value


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    vector = [random.randint(-10, 10) for _ in range(15)]
    print(f"Vector: {vector}")
    first_agent, second_agent = RandomAgent(),MinMaxAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")


if __name__ == "__main__":
    main()
