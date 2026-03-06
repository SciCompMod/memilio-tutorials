
add_executable(exercise1 exercise1.cpp)
target_link_libraries(exercise1 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise2 exercise2.cpp)
target_link_libraries(exercise2 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise10 exercise10.cpp)
target_link_libraries(exercise10 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise11 exercise11.cpp)
target_link_libraries(exercise11 PRIVATE memilio ode_secir Boost::filesystem)

add_executable(exercise_ide exercise_ide.cpp)
target_link_libraries(exercise_ide PRIVATE memilio ide_secir)

add_executable(exercise_lct exercise_lct.cpp)
target_link_libraries(exercise_lct PRIVATE memilio lct_secir)
target_include_directories(exercise_lct PRIVATE ${CMAKE_SOURCE_DIR})
