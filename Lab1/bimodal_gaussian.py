import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Параметри для простої Гаусіанської функції
mean = 0
sigma = 1
x = np.linspace(-10, 10, 400)
membership_simple = fuzz.gaussmf(x, mean, sigma)

# Параметри для двосторонньої Гаусіанської функції
mean1 = -3
sigma1 = 1
mean2 = 3
sigma2 = 1
membership_bi = fuzz.gauss2mf(x, mean1, sigma1, mean2, sigma2)

# Побудова графіків
plt.figure(figsize=(12, 6))

# Графік простої Гаусіанської функції
plt.subplot(1, 2, 1)
plt.plot(x, membership_simple, label='Проста Гаусіанська функція', color='blue')
plt.title('Проста Гаусіанська функція')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')
plt.legend()
plt.grid(True)

# Графік двосторонньої Гаусіанської функції
plt.subplot(1, 2, 2)
plt.plot(x, membership_bi, label='Двостороння Гаусіанська функція', color='red')
plt.title('Двостороння Гаусіанська функція')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
