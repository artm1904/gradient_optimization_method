#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Абстрактный базовый класс для градиентных методов
class GradientDescent {
 public:
  // Чисто виртуальная функция для расчета градиента
  virtual std::vector<double> compute_gradient(const std::vector<double>& x) = 0;

  // Чисто виртуальная функция для обновления значения x
  virtual std::vector<double> update_x(const std::vector<double>& x,
                                       const std::vector<double>& gradient,
                                       double learning_rate) = 0;

  // Виртуальная функция для вычисления значения целевой функции
  virtual double objective_function(const std::vector<double>& x);

  // Основной метод градиентного спуска
  std::vector<double> solve(std::vector<double> initial_x, double learning_rate, double tolerance,
                            int max_iterations);

  // Метод для получения количества итераций
  int get_iteration_count() const { return iteration_count_; }

  // Метод для вывода результатов (теперь в абстрактном классе)
  void print_results(const std::vector<double>& solution);

 protected:
  int iteration_count_ = 0;      // Счетчик итераций.
  bool found_solution_ = false;  // Флаг, указывающий, найдено ли решение.
};
