import copy
import os
import pickle
import pygame
import time
import numpy as np

from food import Food
from model import game_state_to_data_sample, import_data, split_data
from model import Kernel, SVM_train_model
from snake import Snake, Direction

from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF


def main():
    pygame.init()
    bounds = (600, 600)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # agent = HumanAgent(block_size, bounds)  # Once your agent is good to go, change this line
    agent = BehavioralCloningAgent(block_size, bounds, "data/snakerun5.pickle")
    scores = []
    run = True
    pygame.time.delay(1000)
    while run:
        pygame.time.delay(80)  # Adjust game speed, decrease to test your agent and model quickly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    agent.dump_data()
    pygame.quit()


class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)  # Last 10 frames are when you press exit, so they are bad, skip them


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds, file_path="data/2023-12-06_173416.pickle"):
        self.block_size = block_size
        self.bounds = bounds
        X, y = import_data(file_path)
        X_test, X_train, y_test, y_train = split_data(X, y)
        kernel = Kernel.radial_basis()
        c = 1e-2
        SVM_tm = SVM_train_model(kernel, c)
        self.SVM_pm = SVM_tm.train(np.array(X_train), np.array(y_train))

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = game_state_to_data_sample(game_state, self.bounds[0], self.bounds[1], self.block_size)
        new_direction = self.SVM_pm.predict(data_sample)
        print(f"{data_sample}, direction:{new_direction}")
        if new_direction == 0:
            action = Direction.DOWN
        elif new_direction == 1:
            action = Direction.UP
        elif new_direction == 2:
            action = Direction.RIGHT
        elif new_direction == 3:
            action = Direction.LEFT
        print(data_sample)
        return action


class BehavioralCloningAgent2:
    def __init__(self, block_size, bounds, file_path="data/2023-12-06_173416.pickle"):
        self.block_size = block_size
        self.x_bound = bounds[0]
        self.y_bound = bounds[1]
        self.data = []
        X, y = import_data(file_path)
        self.svm = SVC(C=1e-2, kernel='poly', degree=4)
        X_test, self.X_train, y_test, self.y_train = split_data(X, y)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        X_pred = game_state_to_data_sample(game_state, self.x_bound, self.y_bound, self.block_size)
        X_pred = [X_pred, X_pred]
        new_direction = self.svm.fit(self.X_train, self.y_train).predict(X_pred[0:1])
        if new_direction == 2:
            action = Direction.DOWN
        elif new_direction == 0:
            action = Direction.UP
        elif new_direction == 1:
            action = Direction.RIGHT
        elif new_direction == 3:
            action = Direction.LEFT

        self.data.append((copy.deepcopy(game_state), action))
        return action


if __name__ == "__main__":
    main()
