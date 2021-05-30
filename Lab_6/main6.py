import math
import random
import time
from _decimal import Decimal
from functools import reduce
from itertools import compress

import numpy
from scipy.stats import f, t


def generate_y(m, factors_table):
    return [[round(func(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)] for row in factors_table]


def func(x1, x2, x3):
    coeffs = [6.7, 6.5, 1.3, 6.5, 2.4, 0.8, 9.6, 2.6, 0.9, 5.8, 2.8]
    return equation_of_regression(x1, x2, x3, coeffs)


def m_ij(*arrays):
    return numpy.average(reduce(lambda accum, el: accum * el, list(map(lambda el: numpy.array(el), arrays))))


def equation_of_regression(x1, x2, x3, cef):
    factors_array = [1, x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2]
    return sum([el[0] * el[1] for el in zip(cef, factors_array)])


def generate_factors_table(raw_array):
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(
        map(lambda x: x ** 2, row)) for row in raw_array]
    return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


def set_factors_table(factors_table):
    def x_i(i):
        with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(factors_table)))
        res = [row[i] for row in with_null_factor]
        return numpy.array(res)

    return x_i


def find_coefficients(factors, y_vals):
    x_i = set_factors_table(factors)
    coeffs = [[m_ij(x_i(column), x_i(row)) for column in range(11)] for row in range(11)]
    y_numpy = list(map(lambda row: numpy.average(row), y_vals))
    free_values = [m_ij(y_numpy, x_i(i)) for i in range(11)]
    beta_coefficients = numpy.linalg.solve(coeffs, free_values)
    return list(beta_coefficients)


def print_matrix(m, N, factors, y_vals, additional_text=":"):
    labels_table = list(map(lambda x: x.ljust(10),
                            ["x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"] + [
                                "y{}".format(i + 1) for i in range(m)]))
    rows_table = [list(factors[i]) + list(y_vals[i]) for i in range(N)]
    print("\nМатриця планування" + additional_text)
    print(" ".join(labels_table))
    print("\n".join([" ".join(map(lambda j: "{:<+10}".format(j), rows_table[i])) for i in range(len(rows_table))]))
    print("\t")


def print_equation(coeffs, importance=[True] * 11):
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    coefficients_to_print = list(compress(coeffs, importance))
    equation = " ".join(
        ["".join(i) for i in zip(list(map(lambda x: f"{x:+.2f}", coefficients_to_print)), x_i_names)])
    print("Рівняння регресії: y = " + equation)


def cochran_criteria(m, N, y_table):
    start_cochrane = time.time()

    def get_cochran_value(f1, f2, q):
        partResult1 = q / f2
        params = [partResult1, f1, (f2 - 1) * f1]
        fisher = f.isf(*params)
        result = fisher / (fisher + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    print("Перевірка за критерієм Кохрена: m = {}, N = {}".format(m, N))
    y_variations = [numpy.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation / sum(y_variations)
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1 - p
    gt = get_cochran_value(f1, f2, q)
    stop = time.time()
    print(f"Gp = {gp}; Gt = {gt}; f1 = {f1}; f2 = {f2}; q = {q:.2f}")
    if gp < gt:
        print("Gp < Gt, отже дисперсії рівномірні, тому все правильно")
        print(f"Час виконання критерія Кохрена: {str(stop - start_cochrane)}")

        return True
    else:
        print("Gp > Gt, отже дисперсії нерівномірні, тому змінюємо значення m")
        return False


def student_criteria(m, N, y_table, beta_coefficients):
    start_student = time.time()

    def get_student_value(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()

    print(f"\nПеревірка за критерієм Стьюдента: m = {m}, N = {N} ")
    average_variation = numpy.average(list(map(numpy.var, y_table)))
    variation_beta_s = average_variation / N / m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = [abs(beta_coefficients[i]) / standard_deviation_beta_s for i in range(len(beta_coefficients))]
    f3 = (m - 1) * N
    q = 0.05
    t_our = get_student_value(f3, q)
    importance = [True if el > t_our else False for el in list(t_i)]
    print(f"Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print(f"Коефіцієнти ts: " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print(f"f3 = {f3}; q = {q}; tтабл = {t_our}")
    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importance_to_print = ["важливий" if i else "неважливий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    print(*to_print, sep="; ")
    print_equation(beta_coefficients, importance)
    stop = time.time()
    print("Час виконання критерія Стьюдента:" + str(stop - start_student))
    return importance


def fisher_criteria(m, N, d, x_table, y_table, b_coefficients, importance):
    start_fisher = time.time()

    def get_fisher_value(f3, f4, q):
        return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()

    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05
    theoretical_y = numpy.array([equation_of_regression(row[0], row[1], row[2], b_coefficients) for row in x_table])
    average_y = numpy.array(list(map(lambda el: numpy.average(el), y_table)))
    s_ad = m / (N - d) * sum((theoretical_y - average_y) ** 2)
    y_variations = numpy.array(list(map(numpy.var, y_table)))
    s_v = numpy.average(y_variations)
    f_p = float(s_ad / s_v)
    f_t = get_fisher_value(f3, f4, q)
    theoretical_values_to_print = list(
        zip(map(lambda x: f"x1 = {x[1]:<10} x2 = {x[2]:<10} x3 = {x[3]:<10}", x_table), theoretical_y))
    print(f"\nПеревірка за критерієм Фішера: m = {m}, N = {N} для таблиці y_table")
    print(f"Теоретичні значення Y для різних комбінацій факторів:")
    print("\n".join([f"{el[0]}: y = {el[1]}" for el in theoretical_values_to_print]))
    print(f"Fp = {f_p}, Ft = {f_t}")
    print(f"Fp < Ft,отже модель адекватна" if f_p < f_t else "Fp > Ft, отже модель неадекватна")
    stop = time.time()
    print(f"Час виконання критерія Фішера: {str(stop - start_fisher)}")
    return True if f_p < f_t else False


def main():
    xmin = [-15, 25, -15]
    xmax = [30, 65, -5]
    norm_plan_raw = [[-1, -1, -1],
                     [-1, +1, +1],
                     [+1, -1, +1],
                     [+1, +1, -1],
                     [-1, -1, +1],
                     [-1, +1, -1],
                     [+1, -1, -1],
                     [+1, +1, +1],
                     [-1.73, 0, 0],
                     [+1.73, 0, 0],
                     [0, -1.73, 0],
                     [0, +1.73, 0],
                     [0, 0, -1.73],
                     [0, 0, +1.73]]

    x0 = [(xmax[_] + xmin[_]) / 2 for _ in range(3)]
    dx = [xmax[_] - x0[_] for _ in range(3)]

    natur_plan_raw = [[xmin[0], xmin[1], xmin[2]],
                      [xmin[0], xmin[1], xmax[2]],
                      [xmin[0], xmax[1], xmin[2]],
                      [xmin[0], xmax[1], xmax[2]],
                      [xmax[0], xmin[1], xmin[2]],
                      [xmax[0], xmin[1], xmax[2]],
                      [xmax[0], xmax[1], xmin[2]],
                      [xmax[0], xmax[1], xmax[2]],
                      [-1.73 * dx[0] + x0[0], x0[1], x0[2]],
                      [1.73 * dx[0] + x0[0], x0[1], x0[2]],
                      [x0[0], -1.73 * dx[1] + x0[1], x0[2]],
                      [x0[0], 1.73 * dx[1] + x0[1], x0[2]],
                      [x0[0], x0[1], -1.73 * dx[2] + x0[2]],
                      [x0[0], x0[1], 1.73 * dx[2] + x0[2]],
                      [x0[0], x0[1], x0[2]]]

    m = 3
    N = 15
    natural_plan = generate_factors_table(natur_plan_raw)
    y_arr = generate_y(m, natur_plan_raw)
    while not cochran_criteria(m, N, y_arr):
        m += 1
        y_arr = generate_y(m, natural_plan)

    print_matrix(m, N, natural_plan, y_arr, " для натуралізованих факторів:")
    coefficients = find_coefficients(natural_plan, y_arr)
    print_equation(coefficients)
    importance = student_criteria(m, N, y_arr, coefficients)
    d = len(list(filter(None, importance)))
    fisher_criteria(m, N, d, natural_plan, y_arr, coefficients, importance)


if __name__ == '__main__':
    main()
