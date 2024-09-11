import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Створюємо масив значень x
x = np.linspace(0, 10, 500)

# Параметри для функцій
a, b = 2, 5
c, d = 6, 9

# Обчислення значень для S-, Z- та PI-функцій
y_s = fuzz.smf(x, a, b)  # S-функція
y_z = fuzz.zmf(x, c, d)  # Z-функція
y_pi = fuzz.pimf(x, a, b, c, d)  # PI-функція

# Побудова графіків
plt.figure(figsize=(10, 6))

# S-функція
plt.subplot(3, 1, 1)
plt.plot(x, y_s, label='S-функція')
plt.title('S-функція')
plt.grid(True)
plt.legend()

# Z-функція
plt.subplot(3, 1, 2)
plt.plot(x, y_z, label='Z-функція', color='orange')
plt.title('Z-функція')
plt.grid(True)
plt.legend()

# PI-функція
plt.subplot(3, 1, 3)
plt.plot(x, y_pi, label='PI-функція', color='green')
plt.title('PI-функція')
plt.grid(True)
plt.legend()

# Відображення графіків
plt.tight_layout()
plt.show()
