#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "scenario_mpc::scenario_mpc" for configuration "Release"
set_property(TARGET scenario_mpc::scenario_mpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scenario_mpc::scenario_mpc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libscenario_mpc.a"
  )

list(APPEND _cmake_import_check_targets scenario_mpc::scenario_mpc )
list(APPEND _cmake_import_check_files_for_scenario_mpc::scenario_mpc "${_IMPORT_PREFIX}/lib/libscenario_mpc.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
