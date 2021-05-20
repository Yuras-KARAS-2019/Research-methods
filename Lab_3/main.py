import numpy as np
import random
import tkinter as tk
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from tkinter.messagebox import showinfo, showerror


def regression(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y


def find_coefficient(x, y_aver, n):
    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n
    my = sum(y_aver) / n
    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n
    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n
    a1 = sum([y_aver[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([y_aver[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([y_aver[i] * x[i][3] for i in range(len(x))]) / n

    X = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
    Y = [my, a1, a2, a3]
    B = [round(i, 2) for i in solve(X, Y)]
    print('\nРівняння регресії')
    print(f'{B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

    return B


def s_kv(y, y_aver, n, m):
    """квадратна дисперсія
    """
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(s)
    return res


def plan_matrix(n, m, x_range, y_min, y_max):
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)
    x_norm = np.array([[1, -1, -1, -1],
                       [1, -1, 1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1],
                       [1, -1, -1, 1],
                       [1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [1, 1, 1, 1]])
    x_norm = x_norm[:len(y)]

    x = np.ones(shape=(len(x_norm), len(x_norm[0])))
    for i in range(len(x_norm)):
        for j in range(1, len(x_norm[i])):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j - 1][0]
            else:
                x[i][j] = x_range[j - 1][1]

    print('\nМатриця планування')
    print(np.concatenate((x, y), axis=1))

    return x, y, x_norm


def cochrens_criterion(y, y_aver, n, m):
    S_kv = s_kv(y, y_aver, n, m)
    Gp = max(S_kv) / sum(S_kv)
    print('\nПеревірка за критерієм Кохрена')
    return Gp


def bs(x, y, y_aver, n):
    """ оцінки коефіцієнтів
    """
    res = [sum(1 * y for y in y_aver) / n]
    for i in range(3):  # 4 - ксть факторів
        b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
        res.append(b)
    return res


def students_criterion(x, y, y_aver, n, m):
    S_kv = s_kv(y, y_aver, n, m)
    s_kv_aver = sum(S_kv) / n

    s_Bs = (s_kv_aver / n / m) ** 0.5  # статиcтична оцінка дисперсії
    Bs = bs(x, y, y_aver, n)
    ts = [abs(B) / s_Bs for B in Bs]

    return ts


def fishers_criterion(y, y_aver, y_new, n, m, d):
    S_ad = m / (n - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(y))])
    S_kv = s_kv(y, y_aver, n, m)
    S_kv_aver = sum(S_kv) / n

    return S_ad / S_kv_aver


def cohren(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)


def main(n: str, m: str) -> None:
    if n == "" or m == "":
        showinfo("Warning", "Введіть бажані значення у відповідні поля")
    try:
        n = int(n)
        m = int(m)
        f1 = m - 1
        f2 = n
        f3 = f1 * f2
        q = 0.05

        x_range = [(-15, 30), (25, 65), (-15, -5)]  # значення за варіантом
        x_aver_max = (30 + 65 - 5) / 3
        x_aver_min = (- 15 + 25 - 15) / 3

        y_max = 200 + int(x_aver_max)
        y_min = 200 + int(x_aver_min)

        student = partial(t.ppf, q=1 - 0.025)  # табличні значення
        t_student = student(df=f3)

        G_kr = cohren(f1, f2)

        x, y, x_norm = plan_matrix(n, m, x_range, y_min, y_max)
        y_aver = [round(sum(i) / len(i), 2) for i in y]

        B = find_coefficient(x, y_aver, n)

        Gp = cochrens_criterion(y, y_aver, n, m)
        print(f'Gp = {Gp}')
        if Gp < G_kr:
            print(f'З ймовірністю {1 - q} дисперсії однорідні.')
        else:
            showinfo("Info", "Необхідно збільшити ксть дослідів")
            m = int(m) + 1
            main(int(n), m)

        ts = students_criterion(x_norm[:, 1:], y, y_aver, n, m)
        print('\nКритерій Стьюдента:\n', ts)
        res = [t for t in ts if t > t_student]
        final_k = [B[ts.index(i)] for i in ts if i in res]
        print('Коефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
            [i for i in B if i not in final_k]))

        y_new = []
        for j in range(n):
            y_new.append(regression([x[j][ts.index(i)] for i in ts if i in res], final_k))

        print(f'\nЗначення "y" з коефіцієнтами {final_k}')
        print(y_new)

        d = len(res)
        f4 = n - d
        F_p = fishers_criterion(y, y_aver, y_new, n, m, d)

        fisher = partial(f.ppf, q=1 - 0.05)
        f_t = fisher(dfn=f4, dfd=f3)  # табличне значення

        print('\nПеревірка адекватності за критерієм Фішера')
        print(f'Fp = {F_p}')
        print(f'F_t = {f_t}')
        if F_p < f_t:
            showinfo("Result", 'Математична модель адекватна експериментальним даним')
        else:
            showinfo("Result", 'Математична модель не адекватна експериментальним даним')
    except:
        showerror("Error", "Введіть числові значення у всі поля")


if __name__ == '__main__':
    Width = 600
    Height = 350
    root = tk.Tk()
    root.title("Лабораторна #3")
    root.geometry("{Width}x{Height}".format(Width=Width, Height=Height))
    root.resizable(width=False, height=False)

    label_a = tk.Label(root, width=20, height=2, text="Кількість вимірів =", font=("Monotype Corsiva", 16),
                       fg="#0F7F0F")
    label_b = tk.Label(root, width=20, height=2, text="Кількість експериментів =", font=("Monotype Corsiva", 16),
                       fg="#0F7F0F")

    label_a.grid(row=2, column=0)
    label_b.grid(row=3, column=0)

    entry_a = tk.Entry(root, bg="gray", width=15, font=("Times", 16))  # за однією й тією ж самою комбінації факторів
    entry_b = tk.Entry(root, bg="gray", width=15, font=("Times", 16))  # кількість експериментів (рядків матриці план.)

    entry_a.grid(row=2, column=1)
    entry_b.grid(row=3, column=1)

    tk.Label(root, width=25, height=4, text='Лабораторна робота №3\nстудента групи ІВ-92\nКубишки Юрія\nВаріант 13',
             font=('Times New Roman', 18), background="white").grid(row=0, column=0)

    tk.Button(root, width=15, height=3, text="Завдання", font=('Roman', 16), background="lightblue",
              activeforeground="#00FAFF",
              activebackground="gray",
              relief='ridge', command=lambda: main(entry_a.get(), entry_b.get())).grid(row=4, column=0)
    root.config(background="White")
    root.mainloop()
