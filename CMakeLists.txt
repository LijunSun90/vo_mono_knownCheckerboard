# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project( vo_mono_knownCheckerboard )

set(OpenCV_DIR /usr/local/share/OpenCV)

find_package( OpenCV 3.4.1 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Declare the executable target built from your sources
add_executable( vo_mono_knownCheckerboard vo_mono_knownCheckerboard.cpp )

# Link your application with OpenCV libraries
target_link_libraries( vo_mono_knownCheckerboard ${OpenCV_LIBS} )
