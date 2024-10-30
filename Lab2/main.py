import numpy as np
import skfuzzy as fuzz
from skfuzzy import control
from matplotlib import pyplot as plt


def func_y(x):
    return np.sin(np.abs(x)) + np.cos(np.abs(x))


def func_z(x, y):
    return x * np.sin(y)


# Створення нечітких змінних
def create_fuzzy_variables(function, start, finish, slices):
    universe = np.linspace(start, finish, 500)
    mx = control.Antecedent(universe, "mx")
    my = control.Antecedent(universe, "my")
    mf = control.Consequent(universe, "mf")
    c = np.linspace(start, finish, slices)
    diff = c[1] - c[0]

    if function == "Trapezia":
        for i, (a, b, c, d) in enumerate(zip(c - diff, c - diff / 4, c + diff / 4, c + diff)):
            mx[f'mx{i + 1}'] = fuzz.trapmf(mx.universe, [a, b, c, d])
            my[f'my{i + 1}'] = fuzz.trapmf(my.universe, [a, b, c, d])

        c_mf = np.linspace(start, finish, int(slices * 1.5))
        diff_mf = c_mf[1] - c_mf[0]
        for i, (a, b, c, d) in enumerate(zip(c_mf - diff_mf, c_mf - diff_mf / 4, c_mf + diff_mf / 4, c_mf + diff_mf)):
            mf[f'mf{i + 1}'] = fuzz.trapmf(mf.universe, [a, b, c, d])

    elif function == "Triangle":
        mx.automf(names=[f"mx{i}" for i in range(1, 7)])
        my.automf(names=[f"my{i}" for i in range(1, 7)])
        mf.automf(names=[f"mf{i}" for i in range(1, 10)])

    elif function == "Gauss":
        for i, c in enumerate(np.linspace(start, finish, slices)):
            mx[f'mx{i + 1}'] = fuzz.gaussmf(mx.universe, c, (finish - start) / 10)
            my[f'my{i + 1}'] = fuzz.gaussmf(my.universe, c, (finish - start) / 10)

        for i, c in enumerate(np.linspace(start, finish, int(slices * 1.5))):
            mf[f'mf{i + 1}'] = fuzz.gaussmf(mf.universe, c, (finish - start) / 13)

    return mx, my, mf


def calculate_output_idx(i, j):
    if i == 1 or j == 1:
        return 1
    elif i == 2:
        return 1 if j < 3 else 2 if j < 6 else 3
    elif i == 3:
        return 1 if j < 3 else 2 if j == 3 else 3 if j < 6 else 4
    elif i == 4:
        return 1 if j < 3 else 3 if j == 3 else 4 if j == 4 else 5 if j == 5 else 6
    elif i == 5:
        return 1 if j == 1 else 2 if j == 2 else j + 1
    elif i == 6:
        return 1 if j == 1 else j + 1 if j < 4 else j + 2 if j < 6 else j + 3
    else:
        return 1


def calculate_output_idx_diagonal(i):
    if i < 3:
        return 1
    elif i == 3:
        return 2
    elif i == 4:
        return 4
    elif i == 5:
        return 6
    else:
        return 9


# Створення правил для всієї матриці (36 правил)
def create_fuzzy_rules(mx, my, mf):
    rules = []
    for i in range(1, 7):
        for j in range(1, 7):
            output_idx = calculate_output_idx(i, j)
            rule = control.Rule(mx[f'mx{i}'] & my[f'my{j}'], mf[f'mf{output_idx}'])
            rule.label = f'Rule {(i * 6 + j) - 6}: If mx{i} and my{j} then mf{output_idx}'
            rules.append(rule)
            print(rule.label)
    return rules


# Створення правил для діагоналі (6 правил)
def create_fuzzy_rules_diagonal(mx, my, mf):
    rules = []
    for i in range(1, 7):
        output_idx = calculate_output_idx_diagonal(i)
        rule = control.Rule(mx[f'mx{i}'] & my[f'my{i}'], mf[f'mf{output_idx}'])
        rule.label = f'Rule {i}: If mx{i} and my{i} then mf{output_idx}'
        rules.append(rule)
        print(rule.label)
    return rules


# Знаходження середньої похибки
def calculate_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main(function, start, finish, slices, label, diagonal=False):
    mx, my, mf = create_fuzzy_variables(function, start, finish, slices)
    if diagonal:
        rules = create_fuzzy_rules_diagonal(mx, my, mf)
        label += " (діагональні правила)"
    else:
        rules = create_fuzzy_rules(mx, my, mf)
        label += " (36 правил)"

    # Управління
    sys = control.ControlSystem(rules)
    sim = control.ControlSystemSimulation(sys)

    x = np.linspace(start, finish, slices * 10)
    y_true = func_y(x)
    z_true = func_z(x, y_true)
    y_pred = np.zeros_like(x)
    z_pred = np.zeros_like(x)

    # Обрахунок значень для кожної точки
    for i in range(len(x)):
        sim.input['mx'] = x[i]
        sim.input['my'] = y_true[i]
        sim.compute()

        # Чи є значення mf
        try:
            y_pred[i] = sim.output['mf']
            z_pred[i] = func_z(x[i], y_pred[i])
        except KeyError:
            print(f"Warning: No output 'mf' found for inputs mx = {x[i]} and my = {y_true[i]}")
            y_pred[i] = np.nan
            z_pred[i] = np.nan

    # Видаляємо NaN для обрахунку похибки
    valid_indices = ~np.isnan(z_pred)
    z_true = z_true[valid_indices]
    z_pred = z_pred[valid_indices]
    rmse_z = calculate_error(z_true, z_pred)

    plt.figure()
    plt.plot(x[valid_indices], z_true, label="Еталонні значення (X_e)", color='blue')
    plt.plot(x[valid_indices], z_pred, label="Нечітка модель (X_d)", color='red', linestyle='--')
    plt.title(f"{label}: Середня похибка: {rmse_z:.2%}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.show()

    print(f"{label} Fault (z): {rmse_z:.2%}")


# Запуск для різних типів функцій
for func_type in ["Triangle", "Trapezia", "Gauss"]:
    main(func_type, 0.7, np.pi / 2, 6, func_type)
    if func_type == 'Gauss':
        main(func_type, 0.7, np.pi / 2, 6, func_type, diagonal=True)
