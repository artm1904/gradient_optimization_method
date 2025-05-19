#include "thirt.hpp"

std::vector<double> NAGGradientDescent::compute_gradient(const std::vector<double>& x) {
  // Целевая функция: f(W) = (w1 + 2.5)^4 + (w2 + 5)^2 + 5*w1^3
  // df/dw1 = 4*(w1 + 2.5)^3 + 15*w1^2
  // df/dw2 = 2*(w2 + 5)

  std::vector<double> gradient(x.size());

  gradient[0] = 4 * std::pow(x[0] + 2.5, 3) + 15 * std::pow(x[0], 2);
  gradient[1] = 2 * (x[1] + 5);

  return gradient;
}

std::vector<double> NAGGradientDescent::update_x(const std::vector<double>& x,
                                                 const std::vector<double>& gradient,
                                                 double learning_rate) {
  std::vector<double> new_x(x.size());
  double gamma = 1 - lambda_;
  double eta = (1 - gamma) * learning_rate;

  // Инициализация V, если это первый вызов
  if (v_.empty()) {
    v_.resize(x.size(), 0.0);
  }

  // Вычисляем W - gamma * V
  std::vector<double> x_lookahead(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x_lookahead[i] = x[i] - gamma * v_[i];
  }

  // Вычисляем градиент в "смотренной вперед" точке
  std::vector<double> gradient_lookahead = compute_gradient(x_lookahead);

  for (size_t i = 0; i < x.size(); ++i) {
    v_[i] = gamma * v_[i] + eta * gradient_lookahead[i];
    new_x[i] = x[i] - v_[i];
  }
  return new_x;
}
