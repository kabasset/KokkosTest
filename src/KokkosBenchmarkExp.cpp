#include "Kokkos_Timer.hpp"
#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <iostream>

int main(int argc, const char* argv[])
{
  Linx::ProgramContext context("Compute the exponential", argc, argv);
  context.named("side", "The side of the square image", 4096);
  context.parse();
  const auto side = context.as<int>("side");

  Linx::Image<float, 2> a("a", side, side);
  Kokkos::Timer timer;
  a.domain().iterate(
      "init",
      KOKKOS_LAMBDA(int i, int j) { a(i, j) = j - i; });
  Kokkos::fence();
  auto init_time = timer.seconds();
  std::cout << "Init: " << init_time << " s" << std::endl;

  timer.reset();
  a.apply(
      "exp",
      KOKKOS_LAMBDA(auto a_i) { return std::exp(a_i); });
  Kokkos::fence();
  auto exp_time = timer.seconds();
  std::cout << "Exp: " << exp_time << " s" << std::endl;

  return 0;
}
