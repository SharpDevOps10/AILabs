import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import gaussmf

# Створення масиву значень x
x = np.linspace(0, 10, 100)

# Функція приналежності для нечіткої множини A
mu_A = gaussmf(x, 5, 1)

# Доповнення нечіткої множини A
mu_A_complement = 1 - mu_A

# Побудова графіків
plt.figure(figsize=(10, 6))

# Графік для функції приналежності A
plt.plot(x, mu_A, label='Membership Function A', color='blue')

# Графік для доповнення A
plt.plot(x, mu_A_complement, label='Complement of A', color='red', linestyle='--')

# Налаштування графіка
plt.xlabel('x')
plt.ylabel('Membership')
plt.title('Set and its Complement')
plt.grid(True)
plt.legend()
plt.show()
