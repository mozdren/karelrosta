cmake_minimum_required(VERSION 3.14)

project( "Measure" )

# dependencies

set( OpenCV_DIR ../opencv/bin/install )
set( OpenCV_DLL_DIR ${OpenCV_DIR}/x86/vc15/bin)

find_package( OpenCV REQUIRED )

# executables and linked libraries

add_executable( measure main.cpp )
target_link_libraries( measure ${OpenCV_LIBS} )

# Required Resources

file( GLOB OPENCV_DLLS
  "${OpenCV_DLL_DIR}/*.dll"
)
file( COPY ${OPENCV_DLLS} DESTINATION "Release/" )
file( COPY ${OPENCV_DLLS} DESTINATION "Debug/" )

file( COPY "resources/lenna.png" DESTINATION "Release/" )
file( COPY "resources/lenna.png" DESTINATION "Debug/" )

set_property( TARGET measure PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(Configuration)" )