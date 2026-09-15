# Unit test harness for NEAT.cpp.
# Follows Mastering CMake "Testing With CMake and CTest":
# a single test-driver executable built with create_test_sourcelist,
# one add_test() per unit, grouped with LABELS and bounded with TIMEOUT.

# Test sources are auto-discovered: every tests/Test*.cpp defines
#   int <BaseName>(int argc, char* argv[])
# where <BaseName> matches the file name without extension. CONFIGURE_DEPENDS
# makes the generator re-run the glob as part of the build, so dropping a new
# test file in needs no CMakeLists edit and no manual reconfigure. RELATIVE is
# required, not just cosmetic: create_test_sourcelist mangles the path it is
# handed into the entry point symbol it calls, so absolute paths would look for
# _Volumes_..._TestFoo instead of TestFoo.
file(GLOB NEATCPP_TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp)
# Glob order is unspecified; sort so ctest runs (and lists) tests reproducibly.
list(SORT NEATCPP_TEST_SOURCES)

create_test_sourcelist(NEATCPP_TEST_DRIVER_SRCS CommonTests.cpp ${NEATCPP_TEST_SOURCES})

add_executable(NEATcppTests ${NEATCPP_TEST_DRIVER_SRCS})
target_link_libraries(NEATcppTests PRIVATE NEAT.cpp)
# The library already exposes src/ as a PUBLIC include dir, but be explicit
# so the driver compiles even if only the shared lib is enabled.
target_include_directories(NEATcppTests PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Absolute path to the on-disk test data dir (fixtures live in source tree).
target_compile_definitions(NEATcppTests PRIVATE
    NEATCPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# --- Per-test registration ------------------------------------------------
# Every test is registered with PASS_REGULAR_EXPRESSION, which guards against a
# test binary that exits 0 without actually running its checks (every test
# prints "Test passed" on success). Labels and timeout default to a fast unit
# test; a test file categorises itself with directive comments at the top, so
# the metadata lives with the test instead of here:
#
#   // CTest-Labels: Evolution;Fast
#   // CTest-Timeout: 600
set(NEATCPP_TEST_LABELS Unit IO Evolution Compliance Fast)
set(NEATCPP_DEFAULT_TEST_LABELS "Unit;Fast")
set(NEATCPP_DEFAULT_TEST_TIMEOUT 30)

foreach(test_src IN LISTS NEATCPP_TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)

    set(test_labels "${NEATCPP_DEFAULT_TEST_LABELS}")
    set(test_timeout ${NEATCPP_DEFAULT_TEST_TIMEOUT})
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${test_src}" test_directives REGEX "^// CTest-")
    foreach(directive IN LISTS test_directives)
        if(directive MATCHES "CTest-Labels:[ \t]*([^ \t\r]+)")
            set(test_labels "${CMAKE_MATCH_1}")
        elseif(directive MATCHES "CTest-Timeout:[ \t]*([0-9]+)")
            set(test_timeout "${CMAKE_MATCH_1}")
        else()
            message(FATAL_ERROR "${test_src}: unknown directive '${directive}' "
                                "(expected CTest-Labels or CTest-Timeout)")
        endif()
    endforeach()

    foreach(label IN LISTS test_labels)
        if(NOT label IN_LIST NEATCPP_TEST_LABELS)
            message(FATAL_ERROR "${test_src}: unknown ctest label '${label}' "
                                "(known labels: ${NEATCPP_TEST_LABELS})")
        endif()
    endforeach()
    if(test_timeout LESS 1)
        message(FATAL_ERROR "${test_src}: CTest-Timeout must be a positive number of seconds")
    endif()

    add_test(NAME ${test_name} COMMAND NEATcppTests ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "${test_labels}"
        TIMEOUT ${test_timeout}
        PASS_REGULAR_EXPRESSION "Test passed")
endforeach()
