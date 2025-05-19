#include "sixth.hpp"

std::vector<double> AdamGradientDescent::compute_gradient(const std::vector<double>& x) {
  // Целевая функция: f(W) = (w1 + 2.5)^4 + (w2 + 5)^2 + 5*w1^3
  // df/dw1 = 4*(w1 + 2.5)^3 + 15*w1^2
  // df/dw2 = 2 * (x[1] + 5)

  std::vector<double> gradient(x.size());

  gradient[0] = 4 * std::pow(x[0] + 2.5, 3) + 15 * std::pow(x[0], 2);
  gradient[1] = 2 * (x[1] + 5);

  return gradient;
}

std::vector<double> AdamGradientDescent::update_x(const std::vector<double>& x,
                                                  const std::vector<double>& gradient,
                                                  double learning_rate) {
  std::vector<double> new_x(x.size());
  double gamma = 1 - lambda_;

  // Инициализация V, G, если это первый вызов
  if (V_.empty()) {
    V_.resize(x.size(), 0.0);
    G_.resize(x.size(), 0.0);
  }

  for (size_t i = 0; i < x.size(); ++i) {
    // Обновление скользящего среднего градиента (V)
    V_[i] = gamma * V_[i] + (1 - gamma) * gradient[i];

    // Обновление скользящего среднего квадрата градиента (G)
    G_[i] = alpha_ * G_[i] + (1 - alpha_) * gradient[i] * gradient[i];

    // Bias correction
    double v_corrected = V_[i] / (1 - std::pow(gamma, iteration_count_ + 1));
    double g_corrected = G_[i] / (1 - std::pow(alpha_, iteration_count_ + 1));

    // Обновление параметров
    new_x[i] = x[i] - learning_rate * v_corrected / (std::sqrt(g_corrected) + epsilon_);
  }
  return new_x;
}
