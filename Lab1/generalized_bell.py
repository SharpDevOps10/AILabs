import numpy as np
import matplotlib.pyplot as plt

# Параметри для узагальненої дзвонової функції
a = 0  # Центр функції
b = 1  # Ширина функції
c = 2  # Параметр контролю нахилу

# Створення масиву значень x
x = np.linspace(-5, 5, 500)


# Обчислення значень узагальненої дзвонової функції
def generalized_bell(x, a, b, c):
    return 1 / (1 + (np.abs(x - a) / b) ** (2 * c))


# Обчислення y для нашої функції приналежності
y = generalized_bell(x, a, b, c)

# Побудова графіка
plt.plot(x, y)
plt.fill_between(x, y, alpha=0.3)
plt.title('Узагальнений дзвін')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.show()
