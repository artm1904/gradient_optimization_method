#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "zero.hpp"

// Класс, реализующий градиентный метод AdaDelta
class AdaDeltaGradientDescent : public GradientDescent {
 public:
  AdaDeltaGradientDescent(double alpha, double epsilon) : alpha_(alpha), epsilon_(epsilon) {}

  // Реализация вычисления градиента для целевой функции
  std::vector<double> compute_gradient(const std::vector<double>& x) override;

  // Реализация обновления значения x (градиентный шаг)
  std::vector<double> update_x(const std::vector<double>& x, const std::vector<double>& gradient,
                               double learning_rate) override;

 private:
  double alpha_;               // Коэффициент забывания
  double epsilon_;             // Для стабильности (предотвращения деления на 0)
  std::vector<double> G_;      // Экспоненциально затухающее среднее квадратов градиентов
  std::vector<double> Delta_;  // Экспоненциально затухающее среднее квадратов изменений параметров
};