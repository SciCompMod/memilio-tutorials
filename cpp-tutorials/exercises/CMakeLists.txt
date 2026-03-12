
add_executable(exercise01 exercise01.cpp)
target_link_libraries(exercise01 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise02 exercise02.cpp)
target_link_libraries(exercise02 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise03 exercise03.cpp)
target_link_libraries(exercise03 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise05 exercise05.cpp)
target_link_libraries(exercise05 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise07 exercise07.cpp)
target_link_libraries(exercise07 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise10 exercise10.cpp)
target_link_libraries(exercise10 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise11 exercise11.cpp)
target_link_libraries(exercise11 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise_ide exercise_ide.cpp)
target_link_libraries(exercise_ide PRIVATE memilio ide_secir)

add_executable(exercise_lct exercise_lct.cpp)
target_link_libraries(exercise_lct PRIVATE memilio lct_secir)
target_include_directories(exercise_lct PRIVATE ${CMAKE_SOURCE_DIR})
