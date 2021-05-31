import math
import random
from _pydecimal import Decimal
from functools import reduce
from itertools import compress
from tkinter.messagebox import showerror
import time
import numpy as np
from scipy.stats import f, t


def generate_factors_table(raw_array):
    return [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]]
            + list(map(lambda x: round(x ** 2, 5), row))
            for row in raw_array]


def x_i(i, raw_factors_table):
    try:
        assert i <= 10
    except:
        raise AssertionError(" i повинно бути <=10")
    with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(raw_factors_table)))
    res = [row[i] for row in with_null_factor]
    return np.array(res)


def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum * el, arrays))


def cochran_criteria(m, N, y_table):
    print(f"\nПеревірка рівномірності дисперсій за критерієм Кохрена: m = {m}, N = {N}: ")
    y_variations = [np.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation / sum(y_variations)
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1 - p
    gt = get_cochran_value(f1, f2, q)
    print(f"Gp = {gp}; Gt = {gt}; f1 = {f1}; f2 = {f2}; q = {q:.2f}")
    if gp < gt:
        print("Gp < Gt => дисперсії рівномірні")
        return True
    else:
        print("Gp > Gt => дисперсії нерівномірні - потрібно додати експерименти")
        return False


def get_cochran_value(f1, f2, q):
    partResult1 = q / f2
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher / (fisher + (f2 - 1))
    return Decimal(result).quantize(Decimal('.0001')).__float__()


def student_criteria(m, N, y_table, beta_coefficients):
    print(f"\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: m = {m}, N = {N}: ")
    average_variation = np.average(list(map(np.var, y_table)))

    variation_beta_s = average_variation / N / m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = np.array([abs(beta_coefficients[i]) / standard_deviation_beta_s for i in range(len(beta_coefficients))])
    f3 = (m - 1) * N
    q = 0.05

    t = get_student_value(f3, q)
    importance = [True if el > t else False for el in list(t_i)]

    print("Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print("Коефіцієнти ts:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print(f"f3 = {f3}; q = {q}; t табл = {t}")

    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importance_to_print = ["важливий" if i else "неважливий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    betas_to_print = list(compress(beta_coefficients, importance))
    print(*to_print, sep="; ")
    equation = " ".join(["".join(i) for i in zip(list(map(lambda x: f"+ {x:.2f}", betas_to_print)), x_i_names)])
    print(f"Рівняння регресії без незначимих членів: y = {equation}")
    return importance


def get_student_value(f3, q):
    return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()


def calculate_theoretical_y(x_table, b_coefficients, importance):
    x_table = [list(compress(row, importance)) for row in x_table]
    b_coefficients = list(compress(b_coefficients, importance))
    y_vals = np.array([sum(map(lambda x, b: x * b, row, b_coefficients)) for row in x_table])
    return y_vals


def fisher_criteria(m, N, d, naturalized_x_table, y_table, b_coefficients, importance):
    global start_cochrane
    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05

    theoretical_y = calculate_theoretical_y(naturalized_x_table, b_coefficients, importance)
    theoretical_values_to_print = list(
        zip(map(lambda x: f"|\tx1 = {x[1]:^8}|\tx2 = {x[2]:^8}|\tx3 = {x[3]:^8}", naturalized_x_table), theoretical_y))

    y_averages = np.array(list(map(np.average, y_table)))
    s_ad = m / (N - d) * (sum((theoretical_y - y_averages) ** 2))
    y_variations = np.array(list(map(np.var, y_table)))
    s_v = np.average(y_variations)
    f_p = float(s_ad / s_v)
    f_t = get_fisher_value(f3, f4, q)

    stop = time.time()
    if (stop - start_cochrane) > 0.011:
        showerror("Error", "модель неадекватна, час пошуку перевищив 0.011 сек.")
        print(f"Час пошуку коефіцієнтів: {str(stop - start_cochrane)}")
        exit(f"модель неадекватна, час пошуку перевищив 0.011 сек.\nЧас пошуку коефіцієнтів: {str(stop - start_cochrane)}")
    print(f"Час пошуку коефіцієнтів: {str(stop - start_cochrane)}")

    print(f"\nПеревірка адекватності моделі за критерієм Фішера: m = {m}, N = {N}")
    print(f"\nТеоретичні значення y для різних комбінацій факторів:")
    print(f"\n".join([f"{i[0]}: y = {i[1]}" for i in theoretical_values_to_print]))
    print(f"Fp = {f_p}, Ft = {f_t}")
    print(f"Fp < Ft, отже модель адекватна" if f_p < f_t else f"Fp > Ft, отже модель неадекватна")
    return True if f_p < f_t else False


def get_fisher_value(f3, f4, q):
    return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()


def main():
    global start_cochrane
    m = 3
    N = 15
    ymin = 195
    ymax = 206

    raw_naturalized_factors_table = [[-2, -10, -3],     [-2, 10, 6],        [-2, 8, -3],        [-2, 8, 6],
                                     [4, -10, -3],      [4, -10, 6],        [4, -8, -3],        [4, 8, 6],
                                     [-2.645, -1, 1.5], [4.645, -1, 1.5],   [1, -11.935, 1.5],  [1, 9.935, 1.5],
                                     [1, -1, -3.967],   [1, -1, 6.967],     [1, -1, 1.5]]

    raw_factors_table = [[-1, -1, -1],      [-1, +1, +1],   [+1, -1, +1],   [+1, +1, -1],
                         [-1, -1, +1],      [-1, +1, -1],   [+1, -1, -1],   [+1, +1, +1],
                         [-1.215, 0, 0],    [+1.215, 0, 0], [0, -1.215, 0], [0, +1.215, 0],
                         [0, 0, -1.215],    [0, 0, +1.215], [0, 0, 0]]

    start_cochrane = time.time()
    factors_table = generate_factors_table(raw_factors_table)
    for row in factors_table:
        print(row)
    naturalized_factors_table = generate_factors_table(raw_naturalized_factors_table)

    y_arr = [[random.randint(ymin, ymax) for _ in range(m)] for _ in range(N)]
    while not cochran_criteria(m, N, y_arr):
        m += 1
        y_arr = [[random.randint(ymin, ymax) for _ in range(m)] for _ in range(N)]

    y_i = np.array([np.average(row) for row in y_arr])

    coefficients = [[m_ij(x_i(column, raw_factors_table) * x_i(row, raw_factors_table)) for column in range(11)] for row
                    in range(11)]

    free_values = [m_ij(y_i, x_i(i, raw_factors_table)) for i in range(11)]

    beta_coefficients = np.linalg.solve(coefficients, free_values)
    print(list(map(int, beta_coefficients)))

    importance = [student_criteria(m, N, y_arr, beta_coefficients)]
    d = importance.count(None)
    fisher_criteria(m, N, d, naturalized_factors_table, y_arr, beta_coefficients, importance)


if __name__ == '__main__':
    main()
