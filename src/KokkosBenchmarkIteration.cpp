#include "Kokkos_Timer.hpp"
#include "Linx/Data/Vector.h"
#include "Linx/Run/ProgramContext.h"

#include <iostream>

int main(int argc, char* argv[])
{
  Linx::ProgramContext context("Sum two images", argc, argv);
  const auto side = 400;
  Linx::Vector<float, -1> a("a", side * side * side);
  Linx::Vector<float, -1> b("b", side * side * side);
  Linx::Vector<float, -1> c("c", side * side * side);
  Kokkos::Timer timer;
  a.iterate(
      "init",
      KOKKOS_LAMBDA(int i) {
        a[i] = i;
        b[i] = 2 * i;
      });
  auto init_time = timer.seconds();
  std::cout << "Init: " << init_time << "s" << std::endl;

  timer.reset();
  c.generate(
      "sum",
      KOKKOS_LAMBDA(auto a_i, auto b_i) { return a_i + b_i; },
      a,
      b);
  auto sum_time = timer.seconds();
  std::cout << "Sum: " << sum_time << "s" << std::endl;

  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i] != 3 * i) {
      std::cout << c[i] << " != " << 3 * i << std::endl;
      return i;
    }
  }
  return 0;
}
