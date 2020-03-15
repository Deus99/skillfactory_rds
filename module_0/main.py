import numpy as np
import random


# Алгоритм игры с ссылкой на функцию прогнозирования чисел
def game_core_v3(number):
    count = 1
    predict = np.random.randint(35, 65)
    num_min = int(1)
    num_max = int(101)
    while number != predict:
        count += 1
        if number > predict:

            num_min = predict
            predict = predict_numbers(num_min, num_max)
        elif number < predict:
            num_max = predict
            predict = predict_numbers(num_min, num_max)
    return (count)


# функция прогнозирования чисел, в данном случае, среднее врифметическое
def predict_numbers(lim_min, lim_max):
    predict_gamb = (lim_max + lim_min) // 2
    return (predict_gamb)


# Запсук игры 1000 раз
def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1, 101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return (score)


# запускаем
score_game(game_core_v3)
