import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import gaussmf


# Функції для логічних операторів
def And(a, b):
    return np.minimum(a, b)


def Or(a, b):
    return np.maximum(a, b)


# Створення масиву значень x
x = np.linspace(0, 10, 100)

# Обчислення значень гауссових функцій з новими параметрами
y1 = gaussmf(x, 4, 1.5)  # Змінено середнє значення і стандартне відхилення
y2 = gaussmf(x, 6, 2.5)  # Змінено середнє значення і стандартне відхилення

# Обчислення результатів для AND і OR
and_result = And(y1, y2)
or_result = Or(y1, y2)

# Побудова графіків
plt.figure(figsize=(10, 6))

# Графік для першої гауссової функції
plt.plot(x, y1, label='Gaussian MF 1 (mean=4, std=1.5)', color='blue')

# Графік для другої гауссової функції
plt.plot(x, y2, label='Gaussian MF 2 (mean=6, std=2.5)', color='orange')

# Графік для кон'юнктивного (AND) оператора
plt.plot(x, and_result, label='AND Operator', color='green')

# Графік для диз'юнктивного (OR) оператора
plt.plot(x, or_result, label='OR Operator', color='red')

# Налаштування графіка
plt.xlabel('x')
plt.ylabel('Membership')
plt.title('Logic Operators with Gaussian Membership Functions')
plt.grid(True)
plt.legend()
plt.show()
