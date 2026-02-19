cmake_minimum_required(VERSION 3.13)

project(memilio-tutorials VERSION 1.0.0)

set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED "20")

if(NOT DEFINED NUM_JOBS_BUILD)
  set(NUM_JOBS_BUILD "1")
endif()

# Executables should be stored in the build/bin/ folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Download MEmilio with the git tag defined in git_tag.cmake.
include(${CMAKE_SOURCE_DIR}/git_tag.cmake)
# If git tag is not set, throw error.
if(NOT DEFINED GIT_TAG_MEMILIO)
  message(FATAL_ERROR "GIT_TAG_MEMILIO is not defined. Please make sure the git_tag.cmake file is correct.")
endif()

# FetchContent to fetch the MEmilio library in the correct version.
include(FetchContent)
  
FetchContent_Declare(
  memilio
  GIT_REPOSITORY https://github.com/SciCompMod/memilio.git
  GIT_TAG ${GIT_TAG_MEMILIO}
)

FetchContent_MakeAvailable(memilio)

# Disable some options for the build.
set(MEMILIO_BUILD_TESTS OFF)
set(MEMILIO_BUILD_EXAMPLES OFF)
set(MEMILIO_BUILD_SIMULATIONS OFF)

# Add the subdirectory for MEmilio build.
add_subdirectory(${memilio_SOURCE_DIR}/cpp ${memilio_BINARY_DIR})

add_executable(tutorial_ide tutorial_ide.cpp)
target_link_libraries(tutorial_ide PRIVATE memilio ide_secir Boost::filesystem)

add_executable(tutorial_lct tutorial_lct.cpp)
target_link_libraries(tutorial_lct PRIVATE memilio lct_secir Boost::filesystem)

add_executable(tutorial_abm tutorial_abm_household1.cpp)
target_link_libraries(tutorial_abm PRIVATE memilio abm Boost::filesystem)

