import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

x = np.linspace(0, 10, 100)
y = fuzz.sigmf(x, 1, 1)
y2 = fuzz.dsigmf(x, 1, 2, 5, 1)
y3 = fuzz.psigmf(x, 1, 2, 5, 1)

plt.plot(x, y, label='Sigmoidal')
plt.plot(x, y2, label='Double Sigmoidal')
plt.plot(x, y3, label='Product Sigmoidal')

plt.xlabel('x')
plt.ylabel('y')

plt.grid(True)

plt.legend(loc='best')

plt.show()
