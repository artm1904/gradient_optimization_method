#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "zero.hpp"

// Класс, реализующий градиентный метод NAG (Nesterov Accelerated Gradient)
class NAGGradientDescent : public GradientDescent {
 public:
  NAGGradientDescent(double lambda) : lambda_(lambda) {}

  // Реализация вычисления градиента для целевой функции
  std::vector<double> compute_gradient(const std::vector<double>& x) override;

  // Реализация обновления значения x (градиентный шаг)
  std::vector<double> update_x(const std::vector<double>& x, const std::vector<double>& gradient,
                               double learning_rate) override;

 
 private:
  double lambda_;          // Коэффициент забывания
  std::vector<double> v_;  // Промежуточный вектор
};