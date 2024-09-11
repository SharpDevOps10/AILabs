import numpy as np
import matplotlib.pyplot as plt

a = 1
b = 3
c = 7
d = 9

x = np.linspace(a - 1, d + 1, 500)


def trapezoid_mf(x, a, b, c, d):
    return np.clip((np.minimum((x - a) / (b - a), (d - x) / (d - c))), 0, 1)


y = trapezoid_mf(x, a, b, c, d)

plt.plot(x, y, label=f'Trapezoid MF: a={a}, b={b}, c={c}, d={d}')
plt.fill_between(x, y, alpha=0.3)
plt.title('Трапецієподібна функція приналежності')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=1)
plt.grid(True)
plt.show()
