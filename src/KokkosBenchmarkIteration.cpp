#include "Kokkos_Timer.hpp"
#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Reduction.h"

#include <iostream>

int main(int argc, const char* argv[])
{
  Linx::ProgramContext context("Sum two images", argc, argv);
  context.named("side", "The side of the cubic image", 400);
  context.parse();
  const auto side = context.as<int>("side");
  Linx::Image<long, 3> a("a", side, side, side);
  Linx::Image<long, 3> b("b", side, side, side);
  Linx::Image<long, 3> c("c", side, side, side);
  Kokkos::Timer timer;
  a.domain().iterate(
      "init",
      KOKKOS_LAMBDA(int i, int j, int k) {
        a(i, j, k) = i;
        b(i, j, k) = 2 * i;
      });
  Kokkos::fence();
  auto init_time = timer.seconds();
  std::cout << "Init: " << init_time << " s" << std::endl;

  timer.reset();
  c.generate(
      "add",
      KOKKOS_LAMBDA(auto a_i, auto b_i) { return a_i + b_i; },
      Linx::as_readonly(a),
      Linx::as_readonly(b));
  Kokkos::fence();
  auto add_time = timer.seconds();
  std::cout << "Add: " << add_time << " s" << std::endl;

  timer.reset();
  auto sum = Linx::sum("sum", Linx::as_readonly(c));
  Kokkos::fence();
  auto sum_time = timer.seconds();
  std::cout << "Sum: " << sum_time << " s (" << sum << ")" << std::endl;

  return 0;
}
