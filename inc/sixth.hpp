#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "zero.hpp"

// Класс, реализующий градиентный метод Adam
class AdamGradientDescent : public GradientDescent {
 public:
  AdamGradientDescent(double lambda, double alpha, double epsilon)
      : lambda_(lambda), alpha_(alpha), epsilon_(epsilon) {}

  // Реализация вычисления градиента для целевой функции
  std::vector<double> compute_gradient(const std::vector<double>& x) override;

  // Реализация обновления значения x (градиентный шаг)
  std::vector<double> update_x(const std::vector<double>& x, const std::vector<double>& gradient,
                               double learning_rate) override;

 private:
  double lambda_;          // Коэффициент забывания для V
  double alpha_;           // Коэффициент забывания для G
  double epsilon_;         // Для стабильности (предотвращения деления на 0)
  std::vector<double> V_;  // Первый момент (скользящее среднее градиентов)
  std::vector<double> G_;  // Второй момент (скользящее среднее квадратов градиентов)
};