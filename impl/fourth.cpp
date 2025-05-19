#include "fourth.hpp"

std::vector<double> RMSPropGradientDescent::compute_gradient(const std::vector<double>& x) {
  // Целевая функция: f(W) = (w1 + 2.5)^4 + (w2 + 5)^2 + 5*w1^3
  // df/dw1 = 4*(w1 + 2.5)^3 + 15*w1^2
  // df/dw2 = 2 * (x[1] + 5)

  std::vector<double> gradient(x.size());

  gradient[0] = 4 * std::pow(x[0] + 2.5, 3) + 15 * std::pow(x[0], 2);
  gradient[1] = 2 * (x[1] + 5);

  return gradient;
}

std::vector<double> RMSPropGradientDescent::update_x(const std::vector<double>& x,
                                                     const std::vector<double>& gradient,
                                                     double learning_rate) {
  std::vector<double> new_x(x.size());
  double gamma = 1 - lambda_;

  // Инициализация G, если это первый вызов
  if (G_.empty()) {
    G_.resize(x.size(), 0.0);
  }

  for (size_t i = 0; i < x.size(); ++i) {
    G_[i] = gamma * G_[i] + (1 - gamma) * gradient[i] * gradient[i];
    new_x[i] = x[i] - learning_rate * gradient[i] / std::sqrt(G_[i] + epsilon_);
  }
  return new_x;
}
