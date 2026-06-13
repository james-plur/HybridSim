cmake_minimum_required(VERSION 3.14)

project(hybridsim
  VERSION 0.1.0
  DESCRIPTION "Hybrid simulation project"
  LANGUAGES CXX)

option(HYBRIDSIM_BUILD_TESTS "Build and register CTest targets" ON)
option(HYBRIDSIM_BUILD_EXAMPLES "Build example programs" ON)
option(HYBRIDSIM_BUILD_PYTHON "Build Python bindings (hybridsim_py)" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

hybridsim_acquire_simcpp20()

add_library(hybridsim INTERFACE)
target_include_directories(hybridsim INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hybridsim INTERFACE fschuetz04::simcpp20)
target_compile_features(hybridsim INTERFACE cxx_std_20)

function(hybridsim_target_options target)
  target_compile_options(${target} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -fcoroutines)
  endif()
endfunction()

add_executable(simcpp20_install_test tests/install_test.cpp)
target_link_libraries(simcpp20_install_test PRIVATE hybridsim)
hybridsim_target_options(simcpp20_install_test)

add_executable(actor_tests tests/actor_tests.cpp)
target_link_libraries(actor_tests PRIVATE hybridsim)
hybridsim_target_options(actor_tests)

add_executable(dynamic_actor_test tests/dynamic_actor_test.cpp)
target_link_libraries(dynamic_actor_test PRIVATE hybridsim)
hybridsim_target_options(dynamic_actor_test)

add_executable(engine_actor_tests tests/engine_actor_tests.cpp)
target_link_libraries(engine_actor_tests PRIVATE hybridsim)
hybridsim_target_options(engine_actor_tests)

if(HYBRIDSIM_BUILD_EXAMPLES)
  add_executable(actor_ping_pong examples/actor_ping_pong.cpp)
  target_link_libraries(actor_ping_pong PRIVATE hybridsim)
  hybridsim_target_options(actor_ping_pong)

  add_executable(engine_actor samples/engine_actor.cpp)
  target_link_libraries(engine_actor PRIVATE hybridsim)
  hybridsim_target_options(engine_actor)
endif()

if(HYBRIDSIM_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  hybridsim_acquire_pybind11()
  pybind11_add_module(hybridsim_py
    src/python/bindings.cpp
    src/python/bindings_common.cpp
    src/python/engine_bindings.cpp)
  target_link_libraries(hybridsim_py PRIVATE hybridsim)
  hybridsim_target_options(hybridsim_py)
endif()

if(HYBRIDSIM_BUILD_TESTS)
  enable_testing()
  add_test(NAME simcpp20_install_test COMMAND simcpp20_install_test)
  add_test(NAME actor_tests COMMAND actor_tests)
  add_test(NAME dynamic_actor_test COMMAND dynamic_actor_test)
  add_test(NAME engine_actor_tests COMMAND engine_actor_tests)
  if(HYBRIDSIM_BUILD_EXAMPLES)
    add_test(NAME actor_ping_pong COMMAND actor_ping_pong)
    add_test(NAME engine_actor COMMAND engine_actor)
  endif()
  if(HYBRIDSIM_BUILD_PYTHON)
    add_test(
      NAME actor_python_tests
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_actor_python.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_test(
      NAME engine_python_tests
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine_python.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
endif()

install(DIRECTORY include/hybridsim DESTINATION include)
if(HYBRIDSIM_BUILD_PYTHON)
  install(TARGETS hybridsim_py LIBRARY DESTINATION lib)
endif()
