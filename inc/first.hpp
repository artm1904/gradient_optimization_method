#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "zero.hpp"

// Класс, реализующий классический градиентный спуск
class ClassicGradientDescent : public GradientDescent {
 public:
  // Реализация вычисления градиента для целевой функции
  std::vector<double> compute_gradient(const std::vector<double>& x) override;

  // Реализация обновления значения x (градиентный шаг)
  std::vector<double> update_x(const std::vector<double>& x, const std::vector<double>& gradient,
                               double learning_rate) override;
};
