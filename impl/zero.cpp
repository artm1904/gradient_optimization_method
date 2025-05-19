#include "zero.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

double GradientDescent::objective_function(const std::vector<double>& x) {
  return std::pow(x[0] + 2.5, 4) + std::pow(x[1] + 5, 2) + 5 * std::pow(x[0], 3);
};

std::vector<double> GradientDescent::solve(std::vector<double> initial_x, double learning_rate,
                                           double tolerance, int max_iterations) {
  std::vector<double> x = initial_x;
  std::vector<double> gradient;
  iteration_count_ = 0;
  std::vector<double> prev_x = x;  // Сохраняем предыдущее значение x для проверки сходимости
  found_solution_ = false;

  while (true) {
    gradient = compute_gradient(x);

    // Выполняем градиентный шаг
    x = update_x(x, gradient, learning_rate);

    // Проверка на сходимость (разница между предыдущими и текущими значениями x)
    double diff_norm = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      diff_norm += (x[i] - prev_x[i]) * (x[i] - prev_x[i]);
    }
    diff_norm = std::sqrt(diff_norm);

    if (diff_norm < tolerance) {
      found_solution_ = true;
      break;
    }
    if (iteration_count_ >= max_iterations) {
      found_solution_ = false;
      break;
    }

    prev_x = x;  // Обновляем предыдущее значение
    iteration_count_++;
  }

  if (!found_solution_) {
   // std::cout << "Solution not found within the maximum number of iterations." << std::endl;
    return initial_x;  // Возвращаем начальную точку, так как решение не найдено
  }
  return x;
}

void GradientDescent::print_results(const std::vector<double>& solution) {
  if (found_solution_) {
    std::cout << "Solution: w1 = " << solution[0] << ", w2 = " << solution[1] << std::endl;
    std::cout << "Objective function value: " << objective_function(solution) << std::endl;
  }
  std::cout << "Number of iterations: " << iteration_count_ << std::endl;
  std::cout << "----------------------------------------------------------------" << std::endl;
}
