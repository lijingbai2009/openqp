set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SOURCES_modules CONFIGURE_DEPENDS "*.F90" "*.f90")

list(APPEND SOURCES ${SOURCES_modules})
set(SOURCES ${SOURCES} PARENT_SCOPE)
