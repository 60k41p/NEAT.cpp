cmake_minimum_required(VERSION 3.16)
project(NEAT.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(NEATCPP_BUILD_SHARED "Build the shared library" ON)
option(NEATCPP_BUILD_STATIC "Build the static library" ON)
option(NEATCPP_ENABLE_TESTS "Build the unit test harness" ON)

set(NEATCPP_SOURCES
    src/Genome.cpp
    src/Innovation.cpp
    src/NeuralNetwork.cpp
    src/Parameters.cpp
    src/PhenotypeBehavior.cpp
    src/Population.cpp
    src/Random.cpp
    src/Species.cpp
    src/Substrate.cpp
    src/Traits.cpp
    src/Utils.cpp)

set(NEATCPP_HEADERS
    src/AssertMacros.h
    src/Genes.h
    src/Genome.h
    src/Innovation.h
    src/NeuralNetwork.h
    src/Parameters.h
    src/PhenotypeBehavior.h
    src/Population.h
    src/Random.h
    src/Species.h
    src/Substrate.h
    src/Traits.h
    src/Utils.h)

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
