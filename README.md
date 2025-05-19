in main.cpp file set up the following paramentrs;

 // Параметры градиентного спуска
  double learning_rate = 0.01;
  double tolerance = 0.000001;
  int max_iterations = 10000;
  double lambda = 0.1;
  double epsilon = 1e-8;
  double alpha = 0.999;

  // Количество запусков для каждого метода
  int num_runs = 1'000;



  usage:

  cmake -S . -B build/

  cmake --build build

  ./build/lr2 > output.txt

  python3 main.py
