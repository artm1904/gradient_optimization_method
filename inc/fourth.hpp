#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "zero.hpp"

// Класс, реализующий градиентный метод RMSProp
class RMSPropGradientDescent : public GradientDescent {
 public:
  RMSPropGradientDescent(double lambda, double epsilon) : lambda_(lambda), epsilon_(epsilon) {}

  // Реализация вычисления градиента для целевой функции
  std::vector<double> compute_gradient(const std::vector<double>& x) override;

  // Реализация обновления значения x (градиентный шаг)
  std::vector<double> update_x(const std::vector<double>& x, const std::vector<double>& gradient,
                               double learning_rate) override;

 private:
  double lambda_;          // Коэффициент забывания
  double epsilon_;         // Для стабильности (предотвращения деления на 0)
  std::vector<double> G_;  // Промежуточный вектор для хранения квадратов градиентов
};