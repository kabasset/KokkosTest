// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXRUN_PROGRAMCONTEXT
#define _LINXRUN_PROGRAMCONTEXT

#include <Kokkos_Core.hpp>

namespace Linx {

class ProgramOptions { // FIXME placeholder
public:

  ProgramOptions(const std::string& description) : m_desc(description) {}
  virtual ~ProgramOptions() {}

private:

  std::string m_desc;
};

/**
 * @brief Program options ans RAII for the Kokkos context.
 */
class ProgramContext : public ProgramOptions {
public:

  ProgramContext() : ProgramContext("") {}

  ProgramContext(const std::string& description) : ProgramOptions(description), m_scope() {}

  ProgramContext(const std::string& description, int& argc, char* argv[]) :
      ProgramOptions(description), m_scope(argc, argv)
  {}

private:

  Kokkos::ScopeGuard m_scope;
};

} // namespace Linx

#endif
