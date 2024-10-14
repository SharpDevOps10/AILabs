import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Генеруємо випадкові дані для клієнтів
# X1: Кількість покупок, X2: Сума покупок, X3: Кількість відвідувань
np.random.seed(42)
clients = np.random.rand(300, 3) * 300  # 300 клієнтів, 3 ознаки

# Транспонування даних для Fuzzy C-Means
data = clients.T

# Застосування алгоритму Fuzzy C-means
n_clusters = 3  # Кількість кластерів
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, c=n_clusters, m=2, error=0.005, maxiter=500, init=None)

# Отримуємо кластери та центри
cluster_membership = np.argmax(u, axis=0)
cluster_centers = cntr

# Виведення результатів
print("Кластерні центри:\n", cluster_centers)

# Побудова графіку початкових даних
plt.figure()
plt.scatter(clients[:, 0], clients[:, 1], c='blue', label='Дані клієнтів')
plt.title('Початкові дані клієнтів')
plt.xlabel('Кількість покупок на місяць')
plt.ylabel('Загальна сума покупок')
plt.show()

# Побудова графіку зміни цільової функції
plt.figure()
plt.plot(jm)
plt.title('Графік зміни значень цільової функції')
plt.xlabel('Ітерація')
plt.ylabel('Значення цільової функції')
plt.show()

# Побудова графіку кластерів після кластеризації
plt.figure()
for i in range(n_clusters):
    plt.scatter(clients[cluster_membership == i, 0], clients[cluster_membership == i, 1], label=f'Кластер {i + 1}')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100, label='Центри кластерів')
plt.title('Кластеризація клієнтів на основі їх поведінки')
plt.xlabel('Кількість покупок на місяць')
plt.ylabel('Загальна сума покупок')
plt.show()

# Розрахунок коефіцієнта розбиття для різної кількості кластерів
coefficients = []
cluster_range = range(2, 10)  # Вибираємо діапазон кількості кластерів
for i in cluster_range:
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, c=i, m=2, error=0.005, maxiter=500, init=None)
    coefficients.append(fpc)
    print(f"Кількість кластерів: {i} - Коефіцієнт розбиття: {round(fpc, 4)}")

# Побудова графіка коефіцієнта розбиття
plt.figure()
plt.plot(cluster_range, coefficients, c='blue', marker='o')
plt.title("Коефіцієнт розбиття залежно від кількості кластерів")
plt.xlabel("Кількість кластерів")
plt.ylabel("Коефіцієнт розбиття")
plt.xticks(cluster_range)
plt.grid()
plt.show()
