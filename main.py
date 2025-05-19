import matplotlib.pyplot as plt
import numpy as np

# Чтение данных из файла
with open('output.txt', 'r') as f:
    lines = f.readlines()

# Извлечение данных для каждого метода
data = []
for i in range(6):
    # Находим начало блока данных для метода i
    start_index = next((index for (index, line) in enumerate(lines) if f"Method {i+1}:" in line), None)

    # Извлекаем данные об итерациях
    if start_index is not None:
        iterations_line = lines[start_index + 2].strip()  # Строка с Average iterations
        q_line = lines[start_index + 3].strip()  # Строка с Q1 и Q3
        min_max_line = lines[start_index + 4].strip()    # Строка с Min и Max
        min_max_line_next = lines[start_index + 5].strip()

        # Выводим отладочную информацию
        print(f"Method {i+1}:")
        print(f"  iterations_line: {iterations_line}")
        print(f"  q_line: {q_line}")
        print(f"  min_max_line: {min_max_line}")
        print(f"  min_max_line_next: {min_max_line_next}")

        # Обработка случая, когда все значения равны
        if "Q1:" not in q_line:
            q1_val = int(float(q_line.split(":")[1].strip()))  # Берем значение из строки Median iterations
            q3_val = q1_val

            min_max_values = min_max_line_next.split(":")[1]

            #Тут вся магия
            min_max_values = min_max_values.split(",")[0].replace("Min:", "").strip()
            min_val = int(min_max_values)
            max_val = int(min_max_values)

        else:
            q1_val = int(q_line.split(":")[1].split(",")[0].strip())
            q3_val = int(q_line.split(":")[1].split(",")[1].strip())
            min_max_values = min_max_line.split(":")[1].split(",")
            min_val = int(min_max_values[0].strip().split(" ")[1])
            max_val = int(min_max_values[1].strip().split(" ")[1])

        # Добавляем в данные
        data.append([min_val, max_val, q1_val, q3_val])

# Создание ящика с усами
plt.figure(figsize=(10, 6))
plt.boxplot(data, tick_labels=[f'Method {i+1}' for i in range(6)]) # Исправлено labels на tick_labels
plt.title('Iterations for Gradient Descent Methods (Box Plots)')
plt.xlabel('Method')
plt.ylabel('Number of Iterations')
plt.grid(True)
plt.savefig('boxplot.png')
plt.show()