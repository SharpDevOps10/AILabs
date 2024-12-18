import numpy as np

# Матричне представлення літер D, O, V, Z
DOVZ_letters = np.array([
    [
        1,
        1, 1, 1,
        1, 1, 1,
        1, -1, -1  # D
    ],
    [
        1,
        1, 1, 1,
        1, -1, 1,
        1, 1, 1  # O
    ],
    [
        1,
        1, -1, 1,
        1, -1, 1,
        1, 1, 1  # V
    ],
    [
        1,
        1, 1, 1,
        -1, -1, 1,
        1, 1, 1  # Z
    ]
])

# Очікуваний вихід для кожної літери
t = np.array([1, 1, 1, 1])


# Функція для оцінки виходу нейрону
def evaluate(x, w, w0):
    return (x * w).sum() + w0 > 0


# Функція навчання, яка коригує ваги на основі правил Хебба
def train(letters, answers):
    weights = np.zeros(10)
    while True:
        for i, letter in enumerate(letters):
            weights = weights + letter * answers[i]
        if all([
            int(evaluate(
                letters[i][1:],
                weights[1:],
                weights[0]
            )) == answers[i]
            for i in range(len(letters))
        ]):
            break
    return weights


# Навчання мережі
trained_weights = train(DOVZ_letters, t)

# Тестові дані для перевірки роботи мережі
test_letters = np.array([
    [
        1, 1, 1,
        1, 1, 1,
        1, -1, -1  # D
    ],
    [
        1, 1, 1,
        1, -1, 1,
        1, 1, 1  # O
    ],
    [
        -1, 1, -1,
        -1, 1, -1,
        -1, 1, -1  # (некоректний) І
    ],
    [
        1, -1, 1,
        1, -1, 1,
        1, 1, 1  # V
    ],
    [
        1, 1, 1,
        -1, -1, 1,
        1, 1, 1  # Z
    ],
])

for idx, letter in enumerate(test_letters):
    result = evaluate(letter, trained_weights[1:], trained_weights[0])
    if result == 1:
        print(f'Відома літера {idx + 1}')
    else:
        print(f'Невідома літера {idx + 1}')
