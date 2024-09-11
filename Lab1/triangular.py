import numpy as np
import matplotlib.pyplot as plt

a = 2
b = 5
c = 8

x = np.linspace(a - 2, c + 2, 500)


def triangle_mf(x, a, b, c):
    return np.clip(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0, 1)


y = triangle_mf(x, a, b, c)

plt.plot(x, y, label=f'Triangle MF: a={a}, b={b}, c={c}')
plt.fill_between(x, y, alpha=0.3)
plt.title('Трикутна функція приналежності')
plt.xlabel('x')
plt.ylabel('Приналежність')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=1)

plt.grid(True)
plt.show()
