cmake_minimum_required(VERSION 3.0.2)
project(aiplan)
set(CMAKE_BUILD_TYPE debug)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  prediction
  roscpp
  rospy
  std_msgs
  wheel
)
add_message_files(
  FILES
  AiPlan.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs prediction
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aiplan
#  CATKIN_DEPENDS prediction roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
add_executable(ai_control src/ai_control.cpp src/ai_odom_nav.cpp)

target_link_libraries(ai_control
  ${catkin_LIBRARIES}
)

add_dependencies(ai_control ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

