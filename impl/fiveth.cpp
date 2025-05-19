#include "fiveth.hpp"

std::vector<double> AdaDeltaGradientDescent::compute_gradient(const std::vector<double>& x) {
  // Целевая функция: f(W) = (w1 + 2.5)^4 + (w2 + 5)^2 + 5*w1^3
  // df/dw1 = 4*(w1 + 2.5)^3 + 15*w1^2
  // df/dw2 = 2 * (x[1] + 5)

  std::vector<double> gradient(x.size());

  gradient[0] = 4 * std::pow(x[0] + 2.5, 3) + 15 * std::pow(x[0], 2);
  gradient[1] = 2 * (x[1] + 5);

  return gradient;
}

std::vector<double> AdaDeltaGradientDescent::update_x(const std::vector<double>& x,
                                                      const std::vector<double>& gradient,
                                                      double learning_rate) {
  // learning_rate не используется, так как AdaDelta - метод без learning rate
  std::vector<double> new_x(x.size());

  // Инициализация G и Delta, если это первый вызов (устанавливаем начальные значения Delta=0.01)
  if (G_.empty()) {
    G_.resize(x.size(), 0.0);
    Delta_.resize(x.size(), 0.01);
  }

  for (size_t i = 0; i < x.size(); ++i) {
    G_[i] = alpha_ * G_[i] + (1 - alpha_) * gradient[i] * gradient[i];
    double delta = gradient[i] * (std::sqrt(Delta_[i]) + epsilon_) / (std::sqrt(G_[i]) + epsilon_);
    Delta_[i] = alpha_ * Delta_[i] + (1 - alpha_) * delta * delta;
    new_x[i] = x[i] - delta;  // Применяем изменение
  }
  return new_x;
}
