// Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#include <Kokkos_Core.hpp>

/**
 * @brief RAII for the Kokkos context.
 */
struct KokkosContext {

  /**
   * @brief Parse `--kokkos-` options
   */
  KokkosContext(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
  }

  /**
   * @brief Initialize default execution and host spaces.
   */
  KokkosContext() {
    Kokkos::initialize(); // Default execution and host spaces
  }

  /**
   * @brief Finalize.
   */
  ~KokkosContext() {
    Kokkos::finalize();
  }
};