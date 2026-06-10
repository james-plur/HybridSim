include_guard(GLOBAL)

set(HYBRIDSIM_SIMCPP20_REPO
    "https://github.com/fschuetz04/simcpp20.git"
    CACHE STRING "Git repository for simcpp20")
set(HYBRIDSIM_SIMCPP20_TAG
    "main"
    CACHE STRING "Git branch or tag for simcpp20")

set(HYBRIDSIM_PYBIND11_REPO
    "https://github.com/pybind/pybind11.git"
    CACHE STRING "Git repository for pybind11")
set(HYBRIDSIM_PYBIND11_TAG
    "v2.13.6"
    CACHE STRING "Git branch or tag for pybind11")

function(_hybridsim_third_party_ready path out_var)
  if(EXISTS "${path}/CMakeLists.txt")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_hybridsim_git_clone name repo tag dest out_var)
  find_package(Git REQUIRED)
  get_filename_component(_parent "${dest}" DIRECTORY)
  file(MAKE_DIRECTORY "${_parent}")

  message(STATUS "hybridsim: cloning ${name} (${tag}) into ${dest}")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch "${tag}" "${repo}" "${dest}"
    RESULT_VARIABLE _result
    ERROR_VARIABLE _stderr
    OUTPUT_QUIET
  )
  if(NOT _result EQUAL 0)
    message(WARNING "hybridsim: git clone ${name} failed:\n${_stderr}")
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(hybridsim_acquire_simcpp20)
  set(_local "${CMAKE_SOURCE_DIR}/third_party/simcpp20")
  set(FSCHUETZ04_SIMCPP20_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FSCHUETZ04_SIMCPP20_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  _hybridsim_third_party_ready("${_local}" _ready)
  if(NOT _ready)
    _hybridsim_git_clone(simcpp20 ${HYBRIDSIM_SIMCPP20_REPO} ${HYBRIDSIM_SIMCPP20_TAG}
                         "${_local}" _cloned)
    _hybridsim_third_party_ready("${_local}" _ready)
  else()
    message(STATUS "hybridsim: using third_party/simcpp20")
  endif()

  if(_ready)
    add_subdirectory("${_local}" "${CMAKE_BINARY_DIR}/third_party/simcpp20")
    return()
  endif()

  message(STATUS "hybridsim: falling back to FetchContent for simcpp20")
  include(FetchContent)
  FetchContent_Declare(
    simcpp20
    GIT_REPOSITORY ${HYBRIDSIM_SIMCPP20_REPO}
    GIT_TAG ${HYBRIDSIM_SIMCPP20_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(simcpp20)
endfunction()

function(hybridsim_acquire_pybind11)
  set(_local "${CMAKE_SOURCE_DIR}/third_party/pybind11")

  _hybridsim_third_party_ready("${_local}" _ready)
  if(NOT _ready)
    _hybridsim_git_clone(pybind11 ${HYBRIDSIM_PYBIND11_REPO} ${HYBRIDSIM_PYBIND11_TAG}
                         "${_local}" _cloned)
    _hybridsim_third_party_ready("${_local}" _ready)
  else()
    message(STATUS "hybridsim: using third_party/pybind11")
  endif()

  if(_ready)
    add_subdirectory("${_local}" "${CMAKE_BINARY_DIR}/third_party/pybind11")
    return()
  endif()

  if(pybind11_DIR)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "hybridsim: using pybind11 from pybind11_DIR=${pybind11_DIR}")
    return()
  endif()

  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    message(STATUS "hybridsim: using installed pybind11 (${pybind11_DIR})")
    return()
  endif()

  message(STATUS "hybridsim: falling back to FetchContent for pybind11")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY ${HYBRIDSIM_PYBIND11_REPO}
    GIT_TAG ${HYBRIDSIM_PYBIND11_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endfunction()
