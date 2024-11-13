import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf

# Налаштування випадкових значень для відтворюваності
np.random.seed(42)

# Генерація випадкових даних для вхідних змінних
data_size = 1000
temperature = np.random.uniform(15, 30, data_size)  # Температура від 15 до 30 градусів
humidity = np.random.uniform(30, 80, data_size)  # Вологість від 30 до 80%
time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], data_size)
activity_type = np.random.choice(['low', 'medium', 'high'], data_size)

# Перетворення категорійних змінних в числовий формат для моделі
time_of_day_num = pd.get_dummies(time_of_day)
activity_type_num = pd.get_dummies(activity_type)

# Формування вибірки в DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'time_of_day_morning': time_of_day_num['morning'],
    'time_of_day_afternoon': time_of_day_num['afternoon'],
    'time_of_day_evening': time_of_day_num['evening'],
    'time_of_day_night': time_of_day_num['night'],
    'activity_low': activity_type_num['low'],
    'activity_medium': activity_type_num['medium'],
    'activity_high': activity_type_num['high'],
})

# Генерація залежної змінної "споживання енергії" з додаванням випадкового шуму
energy_consumption = 0.5 * temperature + 0.3 * humidity + \
                     2 * time_of_day_num['morning'] + \
                     3 * time_of_day_num['afternoon'] + \
                     4 * time_of_day_num['evening'] + \
                     1.5 * time_of_day_num['night'] + \
                     1.5 * activity_type_num['low'] + \
                     2.5 * activity_type_num['medium'] + \
                     3.5 * activity_type_num['high'] + \
                     np.random.normal(0, 0.5, data_size)

df['energy_consumption'] = energy_consumption

# Нечіткі змінні
temp = ctrl.Antecedent(np.arange(15, 31, 1), 'temperature')
hum = ctrl.Antecedent(np.arange(30, 81, 1), 'humidity')
energy = ctrl.Consequent(np.arange(0, 20, 1), 'energy')

# Функції належності
temp['low'] = fuzz.trimf(temp.universe, [15, 15, 22])
temp['medium'] = fuzz.trimf(temp.universe, [15, 22, 30])
temp['high'] = fuzz.trimf(temp.universe, [22, 30, 30])

hum['low'] = fuzz.trimf(hum.universe, [30, 30, 55])
hum['medium'] = fuzz.trimf(hum.universe, [30, 55, 80])
hum['high'] = fuzz.trimf(hum.universe, [55, 80, 80])

energy['low'] = fuzz.trimf(energy.universe, [0, 0, 10])
energy['medium'] = fuzz.trimf(energy.universe, [5, 10, 15])
energy['high'] = fuzz.trimf(energy.universe, [10, 20, 20])

# Правила нечіткої логіки
rule1 = ctrl.Rule(temp['low'] & hum['low'], energy['low'])
rule2 = ctrl.Rule(temp['medium'] & hum['medium'], energy['medium'])
rule3 = ctrl.Rule(temp['high'] & hum['high'], energy['high'])

energy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
energy_sim = ctrl.ControlSystemSimulation(energy_ctrl)


# Обчислення нечітких виходів і додавання їх як нових ознак
def compute_fuzzy_output(temp_value, hum_value):
    energy_sim.input['temperature'] = temp_value
    energy_sim.input['humidity'] = hum_value
    energy_sim.compute()
    return energy_sim.output['energy']


df['fuzzy_energy'] = df.apply(lambda x: compute_fuzzy_output(x['temperature'], x['humidity']), axis=1)

# Формування вибірки
X = df.drop(columns=['energy_consumption'])
y = df['energy_consumption']

# Поділ на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нейронна мережа
model = Sequential([
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# Компіляція моделі
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Навчання моделі
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Передбачення та обчислення помилки
y_pred = model.predict(X_test)
model.summary()
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Графік втрат під час навчання
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
