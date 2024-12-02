import numpy as np
import tensorflow as tf
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


# Функція, яку треба моделювати
def target_function(x, y):
    return x * np.cos(y) + np.sin(x)


# 1. Побудова нейронної мережі
def build_network():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),  # 2 нейрони вхідного шару
        tf.keras.layers.Dense(4, activation='relu'),  # 4 нейрони
        tf.keras.layers.Dense(8, activation='relu'),  # 8 нейронів
        tf.keras.layers.Dense(8, activation='relu'),  # 8 нейронів
        tf.keras.layers.Dense(6, activation='relu'),  # 6 нейронів
        tf.keras.layers.Dense(4, activation='relu'),  # 4 нейрони
        tf.keras.layers.Dense(1)  # 1 нейрон вихідного шару (без активації)
    ])

    return model


# 2. Функція для оцінки
def evaluate_individual(individual, model, x, y, z_true):
    # Призначення ваг у мережу
    start = 0
    for layer in model.layers:
        weights = layer.get_weights()
        w_shape = weights[0].shape
        b_shape = weights[1].shape

        num_w = np.prod(w_shape)
        num_b = np.prod(b_shape)

        new_weights = np.array(individual[start:start + num_w]).reshape(w_shape)
        new_biases = np.array(individual[start + num_w:start + num_w + num_b]).reshape(b_shape)

        layer.set_weights([new_weights, new_biases])
        start += num_w + num_b

    # Прогноз
    z_pred = model.predict(np.c_[x, y], verbose=0)
    return np.mean((z_true - z_pred.flatten()) ** 2),


# 3. Генетичний алгоритм
def genetic_algorithm(x, y, z_true, model):
    # Ініціалізація DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    num_weights = sum(
        np.prod(layer.get_weights()[0].shape) + np.prod(layer.get_weights()[1].shape) for layer in model.layers)

    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, model=model, x=x, y=y, z_true=z_true)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=80)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=100, stats=stats, halloffame=hof,
                                   verbose=True)

    return pop, log, hof


# 4. Основний блок
if __name__ == "__main__":
    # Генерація даних з меншою кількістю точок
    x = np.linspace(-3, 3, 10)  # Менше точок
    y = np.linspace(-3, 3, 10)
    x, y = np.meshgrid(x, y)
    z_true = target_function(x, y)

    x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z_true.flatten()

    # Створення моделі
    model = build_network()

    # Масштабування z_true
    z_min, z_max = z_true.min(), z_true.max()
    z_flat_scaled = (z_flat - z_min) / (z_max - z_min)

    # Запуск генетичного алгоритму
    _, log, hof = genetic_algorithm(x_flat, y_flat, z_flat_scaled, model)

    # Масштабування передбаченого результату
    z_pred = model.predict(np.c_[x_flat, y_flat], verbose=0).reshape(x.shape)
    z_pred_rescaled = z_pred * (z_max - z_min) + z_min

    # Візуалізація
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, z_true, cmap='viridis', alpha=0.8)
    ax.set_title("True Function")

    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(x, y, z_pred_rescaled, cmap='viridis', alpha=0.8)
    ax.set_title("Predicted Function")

    plt.show()
