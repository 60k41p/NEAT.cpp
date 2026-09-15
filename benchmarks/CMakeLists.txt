# Benchmark harness for NEAT.cpp (see RESULTS.md).
# Not registered as CTest tests: they are timing-based, not pass/fail.
# All *.cpp here build into the one NEATcppBench harness (auto-discovered, so
# new benchmark translation units need no CMakeLists edit; exactly one of them
# provides main()).
file(GLOB NEATCPP_BENCHMARK_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(SORT NEATCPP_BENCHMARK_SOURCES)
add_executable(NEATcppBench ${NEATCPP_BENCHMARK_SOURCES})
target_link_libraries(NEATcppBench PRIVATE NEAT.cpp)
target_include_directories(NEATcppBench PRIVATE ${PROJECT_SOURCE_DIR}/src)
