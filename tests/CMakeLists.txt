# Unit test harness for NEAT.cpp.
# Follows Mastering CMake "Testing With CMake and CTest":
# a single test-driver executable built with create_test_sourcelist,
# one add_test() per unit, grouped with LABELS and bounded with TIMEOUT.

# The individual test sources. Each file defines
#   int <BaseName>(int argc, char* argv[])
# where <BaseName> matches the file name without extension.
set(NEATCPP_TEST_SOURCES
    TestUtils.cxx
    TestRandom.cxx
    TestTraitsGenes.cxx
    TestSubstrate.cxx
    TestInnovation.cxx
    TestParameters.cxx
    TestNeuralNetwork.cxx
    TestGenome.cxx
    TestSpecies.cxx
    TestPopulation.cxx
    TestCompliance.cxx
)

create_test_sourcelist(NEATCPP_TEST_DRIVER_SRCS CommonTests.cxx ${NEATCPP_TEST_SOURCES})

add_executable(NEATcppTests ${NEATCPP_TEST_DRIVER_SRCS})
target_link_libraries(NEATcppTests PRIVATE NEAT.cpp)
# The library already exposes src/ as a PUBLIC include dir, but be explicit
# so the driver compiles even if only the shared lib is enabled.
target_include_directories(NEATcppTests PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Absolute path to the on-disk test data dir (fixtures live in source tree).
target_compile_definitions(NEATcppTests PRIVATE
    NEATCPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Register one CTest test per source, driven through the single executable.
foreach(test_src IN LISTS NEATCPP_TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_test(NAME ${test_name} COMMAND NEATcppTests ${test_name})
endforeach()

# --- Test properties -----------------------------------------------------
# PASS_REGULAR_EXPRESSION guards against a test binary that exits 0 without
# actually running its checks (every test prints "Test passed" on success).
set(NEATCPP_UNIT_TESTS
    TestUtils
    TestRandom
    TestTraitsGenes
    TestSubstrate
    TestInnovation
)
set(NEATCPP_IO_TESTS
    TestParameters
    TestNeuralNetwork
)
set(NEATCPP_EVOLUTION_TESTS
    TestGenome
    TestSpecies
    TestPopulation
)
set(NEATCPP_COMPLIANCE_TESTS
    TestCompliance
)

foreach(t IN LISTS NEATCPP_UNIT_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Unit;Fast"
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS NEATCPP_IO_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Unit;IO;Fast"
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS NEATCPP_EVOLUTION_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Evolution;Fast"
        TIMEOUT 120
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS NEATCPP_COMPLIANCE_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Compliance;Fast"
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()
