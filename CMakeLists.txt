cmake_minimum_required(VERSION 3.16)
project(NEAT.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(NEATCPP_BUILD_SHARED "Build the shared library" ON)
option(NEATCPP_BUILD_STATIC "Build the static library" ON)
option(NEATCPP_ENABLE_TESTS "Build the unit test harness" ON)
option(NEATCPP_ENABLE_BENCHMARKS "Build the benchmark harness" OFF)

# Library sources and headers are auto-discovered. CONFIGURE_DEPENDS makes the
# generator re-run the globs as part of the build, so adding or removing a file
# needs no CMakeLists edit and no manual reconfigure.
file(GLOB NEATCPP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB NEATCPP_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
# Glob order is unspecified; sort for reproducible build graphs.
list(SORT NEATCPP_SOURCES)
list(SORT NEATCPP_HEADERS)

set(NEATCPP_TARGETS "")

if(NEATCPP_BUILD_STATIC)
    add_library(NEAT.cpp STATIC ${NEATCPP_SOURCES} ${NEATCPP_HEADERS})
    set_target_properties(NEAT.cpp PROPERTIES OUTPUT_NAME NEAT)
    list(APPEND NEATCPP_TARGETS NEAT.cpp)
endif()

if(NEATCPP_BUILD_SHARED)
    add_library(NEAT.cpp_shared SHARED ${NEATCPP_SOURCES} ${NEATCPP_HEADERS})
    set_target_properties(NEAT.cpp_shared PROPERTIES OUTPUT_NAME NEAT)
    list(APPEND NEATCPP_TARGETS NEAT.cpp_shared)
endif()

foreach(target IN LISTS NEATCPP_TARGETS)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include/NEAT.cpp>)

    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PUBLIC pthread)
    endif()
endforeach()

# Installation
include(GNUInstallDirs)

install(TARGETS ${NEATCPP_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${NEATCPP_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NEAT.cpp)

# Testing (see Mastering CMake: Testing With CMake and CTest).
# include(CTest) provides BUILD_TESTING and dashboard support.
include(CTest)
if(BUILD_TESTING AND NEATCPP_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(NEATCPP_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
