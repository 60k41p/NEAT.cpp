# Benchmark harness for NEAT.cpp (see BENCHMARKS.md).
# Not registered as CTest tests: they are timing-based, not pass/fail.
add_executable(NEATcppBench Benchmarks.cxx)
target_link_libraries(NEATcppBench PRIVATE NEAT.cpp)
target_include_directories(NEATcppBench PRIVATE ${PROJECT_SOURCE_DIR}/src)
