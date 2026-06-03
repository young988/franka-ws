#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "motion_plan::motion_plan" for configuration ""
set_property(TARGET motion_plan::motion_plan APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(motion_plan::motion_plan PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmotion_plan.so"
  IMPORTED_SONAME_NOCONFIG "libmotion_plan.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS motion_plan::motion_plan )
list(APPEND _IMPORT_CHECK_FILES_FOR_motion_plan::motion_plan "${_IMPORT_PREFIX}/lib/libmotion_plan.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
