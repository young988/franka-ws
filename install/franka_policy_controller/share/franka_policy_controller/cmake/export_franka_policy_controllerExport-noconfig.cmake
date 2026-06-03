#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "franka_policy_controller::franka_policy_controller" for configuration ""
set_property(TARGET franka_policy_controller::franka_policy_controller APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(franka_policy_controller::franka_policy_controller PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libfranka_policy_controller.so"
  IMPORTED_SONAME_NOCONFIG "libfranka_policy_controller.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS franka_policy_controller::franka_policy_controller )
list(APPEND _IMPORT_CHECK_FILES_FOR_franka_policy_controller::franka_policy_controller "${_IMPORT_PREFIX}/lib/libfranka_policy_controller.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
