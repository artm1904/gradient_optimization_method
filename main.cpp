#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>

#include "first.hpp"
#include "fiveth.hpp"
#include "fourth.hpp"
#include "second.hpp"
#include "sixth.hpp"
#include "thirt.hpp"
#include "zero.hpp"

// Функция для вычисления медианы
double median(std::vector<int> &v) {
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  double median = v[n];
  if (v.size() % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + n - 1, v.end());
    median = (median + v[n - 1]) / 2.0;
  }
  return median;
}

// Функция для вычисления квартилей
std::vector<double> quartiles(std::vector<int> &v) {
  std::sort(v.begin(), v.end());
  auto m1 = std::vector<int>(v.begin(), v.begin() + v.size() / 2);
  auto m3 = std::vector<int>(v.begin() + (v.size() + 1) / 2, v.end());
  double q1 = median(m1);
  double q3 = median(m3);
  return {q1, q3};
}

int main() {

 auto start_time = std::chrono::high_resolution_clock::now();

  // Параметры градиентного спуска
  double learning_rate = 0.01;
  double tolerance = 0.000001;
  int max_iterations = 10000;
  double lambda = 0.1;
  double epsilon = 1e-8;
  double alpha = 0.999;

  // Количество запусков для каждого метода
  int num_runs = 1'000;

  // Инициализация начальной точки (случайными числами)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-10, 10);  // Диапазон случайных чисел

  std::vector<double> initial_x = {distrib(gen), distrib(gen)};

  // Создаем экземпляр класса ClassicGradientDescent
  ClassicGradientDescent gd;

  // Создаем экземпляр класса MomentumGradientDescent
  MomentumGradientDescent momentum_gd(lambda);

  // Создаем экземпляр класса NesterovGradientDescent
  NAGGradientDescent nesterov_gd(lambda);

  // Создаем экзменляр класса RMSPropGradientDescent
  RMSPropGradientDescent rmsprop_gd(lambda, epsilon);

  // Создаем экземпляр класса AdaDeltaGradientDescent
  AdaDeltaGradientDescent adam_gd(alpha, epsilon);

  // Создаем массив указателей на объекты класса AdamGradientDescent
  AdamGradientDescent adam_gd_2(lambda, alpha, epsilon);

  std::vector<GradientDescent *> gd_ponters;
  gd_ponters.push_back(&gd);
  gd_ponters.push_back(&momentum_gd);
  gd_ponters.push_back(&nesterov_gd);
  gd_ponters.push_back(&rmsprop_gd);
  gd_ponters.push_back(&adam_gd);
  gd_ponters.push_back(&adam_gd_2);

  // for (int i = 0; i < 6; ++i) {
  //   // Решаем задачу оптимизации
  //   std::vector<double> solution =
  //       gd_ponters[i]->solve(initial_x, learning_rate, tolerance, max_iterations);

  //   // Выводим результаты (используем метод из базового класса)
  //   std::cout << "Solution for method " << i + 1 << ":" << std::endl;
  //   gd_ponters[i]->print_results(solution);
  // }

  // Вектор для хранения данных об итерациях для каждого метода
  std::vector<std::vector<int>> iterations_data(gd_ponters.size());

  // Запускаем каждый метод num_runs раз
  for (int i = 0; i < gd_ponters.size(); ++i) {
    for (int j = 0; j < num_runs; ++j) {
      // Инициализация начальной точки (случайными числами) для каждого запуска
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> distrib(-10, 10);  // Диапазон случайных чисел
      std::vector<double> initial_x = {distrib(gen), distrib(gen)};

      // Решаем задачу оптимизации
      std::vector<double> solution =
          gd_ponters[i]->solve(initial_x, learning_rate, tolerance, max_iterations);

      // Сохраняем количество итераций
      iterations_data[i].push_back(gd_ponters[i]->get_iteration_count());
    }
  }

  // Выводим статистику (для примера, можно вывести в формате, подходящем для построения ящика с усами)
    for (int i = 0; i < gd_ponters.size(); ++i) {
        std::cout << "Method " << i + 1 << ":" << std::endl;
        std::cout << "  Number of runs: " << num_runs << std::endl;

        // Вычисление статистик
        double avg = std::accumulate(iterations_data[i].begin(), iterations_data[i].end(), 0.0) / iterations_data[i].size();
        std::sort(iterations_data[i].begin(), iterations_data[i].end());
        double med = median(iterations_data[i]);
        std::vector<double> qs = quartiles(iterations_data[i]);
        double q1 = qs[0];
        double q3 = qs[1];

        std::cout << "  Average iterations: " << avg << std::endl;
        std::cout << "  Median iterations: " << med << std::endl;
        std::cout << "  Q1: " << q1 << ", Q3: " << q3 << std::endl;

        //  Для ящика с усами также нужны min и max значения
        int min_val =  *std::min_element(iterations_data[i].begin(), iterations_data[i].end());
        int max_val = *std::max_element(iterations_data[i].begin(), iterations_data[i].end());
        std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;

        std::cout << std::endl;
    }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cerr << "Время выполнения программы: " << duration.count() << " мс" << std::endl;


  return 0;
}