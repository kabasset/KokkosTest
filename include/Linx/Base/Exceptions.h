// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_EXCEPTIONS_H
#define _LINXBASE_EXCEPTIONS_H

#include <exception>
#include <string>
#include <utility>

namespace Linx {

/**
 * @ingroup exceptions
 * @brief Base of all exceptions thrown directly by the library.
 */
class Exception : public std::exception {
public:

  /**
   * @brief Destructor.
   */
  virtual ~Exception() = default;

  /**
   * @brief Constructor.
   * @param message Error message
   */
  explicit Exception(const std::string& message) : Exception("Linx error", message) {}

  /**
   * @brief Constructor.
   * @param prefix Error prefix
   * @param message Error message
   */
  explicit Exception(const std::string& prefix, const std::string& message) :
      std::exception(), m_prefix(prefix), m_message(m_prefix + ": " + message)
  {}

  /**
   * @brief Output message.
   */
  const char* what() const noexcept override
  {
    return m_message.c_str();
  }

  /**
   * @brief Append a given line to the message.
   * @param line The line to be appended
   * @param indent Some indentation level
   */
  Exception& append(const std::string& line, std::size_t indent = 0)
  {
    m_message += "\n";
    for (std::size_t i = 0; i < indent; ++i) {
      m_message += "  ";
    }
    m_message += line;
    return *this;
  }

private:

  const std::string m_prefix;
  std::string m_message;
};

/**
 * @ingroup exceptions
 * @brief Exception thrown when trying to read a null pointer.
 */
class NullPtrDereferencing : public Exception {
public:

  /**
   * @brief Constructor.
   */
  NullPtrDereferencing(const std::string& message) : Exception("Null pointer dereferencing", message) {}

  /**
   * @brief Throw if a given pointer is null.
   */
  void may_throw(const void* ptr, const std::string& message)
  {
    if (not ptr) {
      throw NullPtrDereferencing(message);
    }
  }
};

/**
 * @brief Exception thrown if containers have bad size.
 */
class SizeMismatch : public Exception {
public:

  /**
   * @brief Constructor.
   */
  SizeMismatch(const std::string& name, auto value, const auto&...) :
      Exception("Size mismatch", name + " size differ from " + std::to_string(value))
  {}

  /**
   * @brief Throw if sizes mismatch.
   */
  static void may_throw(const std::string& name, auto value, const auto&... containers)
  {
    bool match = ((value == std::size(containers)) && ...);
    if (not match) {
      throw SizeMismatch(name, value, containers...);
    }
  }
};

/**
 * @brief Exception thrown if a value lies out of given bounds.
 * 
 * @tparam Lower The type of lower bound (either `'['` or `'('`)
 * @tparam Upper The type of the upper bound (either `']'` or `')'`)
 * 
 * Example usage:
 * 
 * \code
 * OutOfBounds<'[', ')'>::may_throw("index", i, {0, size});
 * \endcode
 */
template <char Lower, char Upper> // FIXME assert possible values
class OutOfBounds : public Exception {
public:

  /**
   * @brief Constructor.
   */
  OutOfBounds(const std::string& name, auto value, const auto(&bounds)[2]) :
      // FIXME swap value and bounds
      Exception(
          "Out of bounds",
          name + " " + std::to_string(value) + " not in " + Lower + std::to_string(bounds[0]) + ", " +
              std::to_string(bounds[1]) + Upper)
  {}

  /**
   * @brief Throw if a value lies out of given bounds.
   */
  static void may_throw(const std::string& name, auto value, const auto(&bounds)[2]) // FIXME swap value and bounds
  {
    if constexpr (Lower == '[') {
      if (value < bounds[0]) {
        throw OutOfBounds(name, value, bounds);
      }
    } else {
      if (value <= bounds[0]) {
        throw OutOfBounds(name, value, bounds);
      }
    }
    if constexpr (Upper == ']') {
      if (value > bounds[1]) {
        throw OutOfBounds(name, value, bounds);
      }
    } else {
      if (value >= bounds[1]) {
        throw OutOfBounds(name, value, bounds);
      }
    }
  }
};

} // namespace Linx

#endif
