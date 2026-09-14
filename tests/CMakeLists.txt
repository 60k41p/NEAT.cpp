# Unit test harness for MultiNEAT.
# Follows Mastering CMake "Testing With CMake and CTest":
# a single test-driver executable built with create_test_sourcelist,
# one add_test() per unit, grouped with LABELS and bounded with TIMEOUT.

# The individual test sources. Each file defines
#   int <BaseName>(int argc, char* argv[])
# where <BaseName> matches the file name without extension.
set(MULTINEAT_TEST_SOURCES
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

create_test_sourcelist(MULTINEAT_TEST_DRIVER_SRCS CommonTests.cxx ${MULTINEAT_TEST_SOURCES})

add_executable(MultiNEATTests ${MULTINEAT_TEST_DRIVER_SRCS})
target_link_libraries(MultiNEATTests PRIVATE MultiNEAT)
# The library already exposes src/ as a PUBLIC include dir, but be explicit
# so the driver compiles even if only the shared lib is enabled.
target_include_directories(MultiNEATTests PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Absolute path to the on-disk test data dir (fixtures live in source tree).
target_compile_definitions(MultiNEATTests PRIVATE
    MULTINEAT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Register one CTest test per source, driven through the single executable.
foreach(test_src IN LISTS MULTINEAT_TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_test(NAME ${test_name} COMMAND MultiNEATTests ${test_name})
endforeach()

# --- Test properties -----------------------------------------------------
# PASS_REGULAR_EXPRESSION guards against a test binary that exits 0 without
# actually running its checks (every test prints "Test passed" on success).
set(MULTINEAT_UNIT_TESTS
    TestUtils
    TestRandom
    TestTraitsGenes
    TestSubstrate
    TestInnovation
)
set(MULTINEAT_IO_TESTS
    TestParameters
    TestNeuralNetwork
)
set(MULTINEAT_EVOLUTION_TESTS
    TestGenome
    TestSpecies
    TestPopulation
)
set(MULTINEAT_COMPLIANCE_TESTS
    TestCompliance
)

foreach(t IN LISTS MULTINEAT_UNIT_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Unit;Fast"
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS MULTINEAT_IO_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Unit;IO;Fast"
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS MULTINEAT_EVOLUTION_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Evolution;Fast"
        TIMEOUT 120
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()

foreach(t IN LISTS MULTINEAT_COMPLIANCE_TESTS)
    set_tests_properties(${t} PROPERTIES
        LABELS "Compliance;Fast"
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()
