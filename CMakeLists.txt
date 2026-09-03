cmake_minimum_required(VERSION 3.5)
project(MultiNEAT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(MULTINEAT_BUILD_SHARED "Build the shared library" ON)
option(MULTINEAT_BUILD_STATIC "Build the static library" ON)

set(MULTINEAT_SOURCES
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

set(MULTINEAT_HEADERS
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

set(MULTINEAT_TARGETS "")

if(MULTINEAT_BUILD_STATIC)
    add_library(MultiNEAT STATIC ${MULTINEAT_SOURCES} ${MULTINEAT_HEADERS})
    list(APPEND MULTINEAT_TARGETS MultiNEAT)
endif()

if(MULTINEAT_BUILD_SHARED)
    add_library(MultiNEAT_shared SHARED ${MULTINEAT_SOURCES} ${MULTINEAT_HEADERS})
    set_target_properties(MultiNEAT_shared PROPERTIES OUTPUT_NAME MultiNEAT)
    list(APPEND MULTINEAT_TARGETS MultiNEAT_shared)
endif()

foreach(target IN LISTS MULTINEAT_TARGETS)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include/MultiNEAT>)

    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PUBLIC pthread)
    endif()
endforeach()

# Installation
include(GNUInstallDirs)

install(TARGETS ${MULTINEAT_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${MULTINEAT_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/MultiNEAT)
