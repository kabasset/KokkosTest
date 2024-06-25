#include "Kokkos_Timer.hpp"
#include "Linx/Data/Image.h"
#include "Linx/Data/Vector.h"
#include "Linx/Run/ProgramContext.h"

#include <iostream>

int main(int argc, char* argv[])
{
  Linx::ProgramContext context("Sum two images", argc, argv);
  const auto side = 400;
  Linx::Image<float, 3> a("a", side, side, side);
  Linx::Image<float, 3> b("b", side, side, side);
  Linx::Image<float, 3> c("c", side, side, side);
  Kokkos::Timer timer;
  a.domain().iterate(
      "init",
      KOKKOS_LAMBDA(int i, int j, int k) {
        a(i, j, k) = i;
        b(i, j, k) = 2 * i;
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

  return 0;
}
