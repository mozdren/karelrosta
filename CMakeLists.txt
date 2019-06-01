cmake_minimum_required(VERSION 3.14)

project( "Measure" )

# dependencies

set( OpenCV_DIR ../opencv/bin/install )
set( OpenCV_DLL_DIR ${OpenCV_DIR}/x86/vc15/bin)

find_package( OpenCV REQUIRED )

# sources

set ( H_FILES
	genalg.h
	identify.h
	matrix.h
	pmfilter.h
)

set ( CPP_FILES
	genalg.cpp
	identify.cpp
	# main.cpp
	matrix.cpp
	# new.cpp
	new2.cpp
	pmfilter.cpp
)

# executables and linked libraries

add_executable( measure ${CPP_FILES} ${H_FILES})
target_link_libraries( measure ${OpenCV_LIBS} )

# Required Resources

file( GLOB OPENCV_DLLS
  "${OpenCV_DLL_DIR}/*.dll"
)
file( COPY ${OPENCV_DLLS} DESTINATION "Release/" )
file( COPY ${OPENCV_DLLS} DESTINATION "Debug/" )

file( COPY "resources/IMG_1952.JPG" DESTINATION "Release/" )
file( COPY "resources/IMG_1952.JPG" DESTINATION "Debug/" )
file( COPY "resources/template.png" DESTINATION "Release/" )
file( COPY "resources/template.png" DESTINATION "Debug/" )

set_property( TARGET measure PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(Configuration)" )