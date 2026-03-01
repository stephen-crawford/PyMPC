cmake_minimum_required(VERSION 3.14)
project(scenario_mpc VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found. Please install it: sudo apt-get install libeigen3-dev")
endif()

# Define the library source files
set(SCENARIO_MPC_SOURCES
    src/dynamics.cpp
    src/mode_weights.cpp
    src/conformal_safety.cpp
    src/hazard_switch_sampling.cpp
    src/risk_directed_bandit.cpp
    src/adaptive_dro_shift.cpp
    src/dual_risk_monitor.cpp
    src/certificate_first.cpp
    src/scenario_compiler.cpp
    src/runtime_assurance.cpp
    src/trajectory_moments.cpp
    src/scenario_sampler.cpp
    src/collision_constraints.cpp
    src/scenario_pruning.cpp
    src/mpc_controller.cpp
    src/reference_path.cpp
    src/contouring_mpc.cpp
    src/optimal_transport_predictor.cpp
    src/qp_solver.cpp
    src/wasserstein_dro.cpp
    src/experiment_harness.cpp
)

# Create the library
add_library(scenario_mpc ${SCENARIO_MPC_SOURCES})
target_include_directories(scenario_mpc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(scenario_mpc PUBLIC Eigen3::Eigen)

# Set compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scenario_mpc PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter
    )
endif()

# Build tests
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE scenario_mpc)

    add_executable(test_safe_horizon_contouring tests/test_safe_horizon_contouring.cpp)
    target_link_libraries(test_safe_horizon_contouring PRIVATE scenario_mpc)

    add_executable(test_optimal_transport tests/test_optimal_transport.cpp)
    target_link_libraries(test_optimal_transport PRIVATE scenario_mpc)

    add_executable(test_ot_predictor_integration tests/test_ot_predictor_integration.cpp)
    target_link_libraries(test_ot_predictor_integration PRIVATE scenario_mpc)

    add_executable(test_ot_dynamics_verification tests/test_ot_dynamics_verification.cpp)
    target_link_libraries(test_ot_dynamics_verification PRIVATE scenario_mpc)

    add_executable(test_epsilon_guarantee tests/test_epsilon_guarantee.cpp)
    target_link_libraries(test_epsilon_guarantee PRIVATE scenario_mpc)

    add_executable(test_paper_figures tests/test_paper_figures.cpp)
    target_link_libraries(test_paper_figures PRIVATE scenario_mpc)

    add_executable(test_uncertainty_limits tests/test_uncertainty_limits.cpp)
    target_link_libraries(test_uncertainty_limits PRIVATE scenario_mpc)

    add_executable(test_sqp_comparison tests/test_sqp_comparison.cpp)
    target_link_libraries(test_sqp_comparison PRIVATE scenario_mpc)

    add_executable(test_mode_coverage tests/test_mode_coverage.cpp)
    target_link_libraries(test_mode_coverage PRIVATE scenario_mpc)

    add_executable(test_results_section tests/test_results_section.cpp)
    target_link_libraries(test_results_section PRIVATE scenario_mpc)

    add_executable(test_statistical_power tests/test_statistical_power.cpp)
    target_link_libraries(test_statistical_power PRIVATE scenario_mpc)

    add_executable(test_dro_framework tests/test_dro_framework.cpp)
    target_link_libraries(test_dro_framework PRIVATE scenario_mpc)

    add_executable(test_conformal_safety tests/test_conformal_safety.cpp)
    target_link_libraries(test_conformal_safety PRIVATE scenario_mpc)

    add_executable(test_hazard_switch_sampling tests/test_hazard_switch_sampling.cpp)
    target_link_libraries(test_hazard_switch_sampling PRIVATE scenario_mpc)

    add_executable(test_risk_directed_bandit tests/test_risk_directed_bandit.cpp)
    target_link_libraries(test_risk_directed_bandit PRIVATE scenario_mpc)

    add_executable(test_adaptive_dro_shift tests/test_adaptive_dro_shift.cpp)
    target_link_libraries(test_adaptive_dro_shift PRIVATE scenario_mpc)

    add_executable(experiment_runner tests/experiment_runner.cpp)
    target_link_libraries(experiment_runner PRIVATE scenario_mpc)

    add_executable(paper_experiment_runner tests/paper_experiment_runner.cpp)
    target_link_libraries(paper_experiment_runner PRIVATE scenario_mpc)

    add_executable(future_sl_experiments ${CMAKE_CURRENT_SOURCE_DIR}/../future_sl_mpc/experiments/run_future_sl_experiments.cpp)
    target_link_libraries(future_sl_experiments PRIVATE scenario_mpc)

    add_executable(future_sl_edge_tuning ${CMAKE_CURRENT_SOURCE_DIR}/../future_sl_mpc/experiments/run_future_sl_edge_and_tuning.cpp)
    target_link_libraries(future_sl_edge_tuning PRIVATE scenario_mpc)

    add_test(NAME CoreTests COMMAND test_core)
    add_test(NAME SafeHorizonContouringTests COMMAND test_safe_horizon_contouring)
    add_test(NAME OptimalTransportTests COMMAND test_optimal_transport)
    add_test(NAME OTPredictorIntegrationTests COMMAND test_ot_predictor_integration)
    add_test(NAME OTDynamicsVerificationTests COMMAND test_ot_dynamics_verification)
    add_test(NAME EpsilonGuaranteeTests COMMAND test_epsilon_guarantee)
    add_test(NAME PaperFigureTests COMMAND test_paper_figures)
    add_test(NAME ConformalSafetyTests COMMAND test_conformal_safety)
    add_test(NAME HazardSwitchSamplingTests COMMAND test_hazard_switch_sampling)
    add_test(NAME RiskDirectedBanditTests COMMAND test_risk_directed_bandit)
    add_test(NAME AdaptiveDROShiftTests COMMAND test_adaptive_dro_shift)
    add_executable(test_dual_risk_monitor tests/test_dual_risk_monitor.cpp)
    target_link_libraries(test_dual_risk_monitor PRIVATE scenario_mpc)

    add_executable(test_certificate_first tests/test_certificate_first.cpp)
    target_link_libraries(test_certificate_first PRIVATE scenario_mpc)

    add_executable(test_scenario_compiler tests/test_scenario_compiler.cpp)
    target_link_libraries(test_scenario_compiler PRIVATE scenario_mpc)

    add_executable(test_runtime_assurance tests/test_runtime_assurance.cpp)
    target_link_libraries(test_runtime_assurance PRIVATE scenario_mpc)

    add_test(NAME DualRiskMonitorTests COMMAND test_dual_risk_monitor)
    add_test(NAME CertificateFirstTests COMMAND test_certificate_first)
    add_test(NAME ScenarioCompilerTests COMMAND test_scenario_compiler)
    add_test(NAME RuntimeAssuranceTests COMMAND test_runtime_assurance)

    # Custom target to run tests
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_core test_safe_horizon_contouring test_optimal_transport test_ot_predictor_integration test_ot_dynamics_verification test_epsilon_guarantee test_paper_figures
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Build examples
if(BUILD_EXAMPLES)
    add_executable(example_mpc examples/example_mpc.cpp)
    target_link_libraries(example_mpc PRIVATE scenario_mpc)

    add_executable(demo_visualization examples/demo_visualization.cpp)
    target_link_libraries(demo_visualization PRIVATE scenario_mpc)

    add_executable(ascii_visualize examples/ascii_visualize.cpp)
    target_link_libraries(ascii_visualize PRIVATE scenario_mpc)
endif()

# Installation
install(TARGETS scenario_mpc
    EXPORT scenario_mpc-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT scenario_mpc-targets
    FILE scenario_mpc-targets.cmake
    NAMESPACE scenario_mpc::
    DESTINATION lib/cmake/scenario_mpc
)

# Generate package config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/scenario_mpc-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/scenario_mpc-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/scenario_mpc-config.cmake"
    INSTALL_DESTINATION lib/cmake/scenario_mpc
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/scenario_mpc-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/scenario_mpc-config-version.cmake"
    DESTINATION lib/cmake/scenario_mpc
)

# Print configuration summary
message(STATUS "")
message(STATUS "scenario_mpc configuration:")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tests:   ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "")
