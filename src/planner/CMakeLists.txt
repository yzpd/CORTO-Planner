cmake_minimum_required(VERSION 3.0.2)
project(planner)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  visualization_msgs
  tf
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED QUIET)
find_package(NLopt REQUIRED)

if(DEFINED ENV{MOSEK_ROOT})
  set(MOSEK_ROOT $ENV{MOSEK_ROOT})
  set(MOSEK_INCLUDE_DIR "${MOSEK_ROOT}/tools/platform/linux64x86/h")
  set(MOSEK_LIB_DIR "${MOSEK_ROOT}/tools/platform/linux64x86/bin")
else()
  message(FATAL_ERROR "MOSEK_ROOT is not set")
endif()

catkin_package(
  INCLUDE_DIRS ../modules
  LIBRARIES planner
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs visualization_msgs tf
)

include_directories(
  ../modules
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${MOSEK_INCLUDE_DIR}
  ${nlopt_INCLUDE_DIRS}
)

link_directories(
  ${MOSEK_LIB_DIR}
  ${PCL_LIBRARY_DIRS}
  ${nlopt_LIBRARY_DIRS}
)

add_library(planner
  ../modules/map/map.cpp
  ../modules/ros_interface/ros_interface.cpp
  ../modules/path_searching/topo_prm.cpp
  ../modules/raycast/raycast.cpp
  ../modules/cubic_spline/cubic_spline.cpp
  ../modules/cubic_spline/arc_length_spline.cpp
  ../modules/optimization/ref_traj_opt.cpp
  ../modules/corridor/corGen.cpp
  ../modules/nmpc/nominal_nmpc_cor.cpp
)

add_executable(planner_node src/CORTO_planner_main.cpp)

target_link_libraries(planner_node
  planner
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  mosek64
  fusion64
  NLopt::nlopt
)