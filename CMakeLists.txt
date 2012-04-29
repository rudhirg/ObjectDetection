cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(OBJECT_DETECTION)
find_package(PCL 1.6 REQUIRED COMPONENTS common io filters sample_consensus segmentation visualization kdtree features surface octree registration keypoints tracking search range_image)
find_package(Boost)
#find_package(OpenCV REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(/usr/include/opencv-2.3.1)
include_directories(./svm-light)

add_library(svmlight_lib STATIC ./svm-light/svm_common.c ./svm-light/svm_learn.c ./svm-light/svm_hideo.c)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#add_executable(ObjectDetection ObjectDetection.cpp)
#add_executable(ObjectTracking ObjectTracking.cpp)
#target_link_libraries(ObjectDetection pcl_common pcl_io pcl_surface pcl_visualization pcl_filters pcl_features pcl_segmentation pcl_tracking pcl_search)
#target_link_libraries(ObjectDetection ${PCL_LIBRARIES} ${Boost_LIBRARIES} svmlight_lib)

#target_link_libraries(ObjectTracking pcl_common pcl_io pcl_surface pcl_visualization pcl_filters pcl_features pcl_segmentation pcl_tracking pcl_search)
#target_link_libraries(ObjectTracking ${PCL_LIBRARIES} ${Boost_LIBRARIES} svmlight_lib)

# we need FindFLANN.cmake 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(${PCL_DEFINITIONS})
#add_definitions(STANDALONE_
find_package(HDF5)
if(HDF5_FOUND)

  find_package(Flann)
  include_directories(${FLANN_INCLUDE_DIRS})

  include_directories(${HDF5_INCLUDE_DIR})

  add_executable(vfh_build_tree vfh_build_tree.cpp)
  target_link_libraries(vfh_build_tree ${PCL_LIBRARIES} ${Boost_LIBRARIES}
                                   ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY})


  add_definitions(-DNOSTANDALONE_DETECTION)
  add_executable(ObjectTracking ObjectTracking.cpp ObjectDetection.cpp KalmanFilter.cpp)
  #add_executable(ObjectTracking ObjectTracking.cpp)
  target_link_libraries(ObjectTracking ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} svmlight_lib)

#  add_executable(ObjectDetection ObjectDetection.cpp)
#  target_link_libraries(ObjectDetection ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} svmlight_lib)

endif(HDF5_FOUND)

SET(CMAKE_CXX_FLAGS "-g -Wall -O0  -lopencv_ml -lopencv_highgui -lopencv_core -lopencv_objdetect -lopencv_video") 
#-lboost_filesystem-mt -lboost_iostreams-mt")
SET(CMAKE_C_FLAGS_DEBUG)

