# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wenchang/projects/ForceRL/gpd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wenchang/projects/ForceRL/gpd/build

# Include any dependencies generated for this target.
include CMakeFiles/gpd_hand.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gpd_hand.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gpd_hand.dir/flags.make

CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o: CMakeFiles/gpd_hand.dir/flags.make
CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o: ../src/gpd/candidate/hand.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wenchang/projects/ForceRL/gpd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o -c /home/wenchang/projects/ForceRL/gpd/src/gpd/candidate/hand.cpp

CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wenchang/projects/ForceRL/gpd/src/gpd/candidate/hand.cpp > CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.i

CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wenchang/projects/ForceRL/gpd/src/gpd/candidate/hand.cpp -o CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.s

# Object files for target gpd_hand
gpd_hand_OBJECTS = \
"CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o"

# External object files for target gpd_hand
gpd_hand_EXTERNAL_OBJECTS =

libgpd_hand.a: CMakeFiles/gpd_hand.dir/src/gpd/candidate/hand.cpp.o
libgpd_hand.a: CMakeFiles/gpd_hand.dir/build.make
libgpd_hand.a: CMakeFiles/gpd_hand.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wenchang/projects/ForceRL/gpd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgpd_hand.a"
	$(CMAKE_COMMAND) -P CMakeFiles/gpd_hand.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpd_hand.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpd_hand.dir/build: libgpd_hand.a

.PHONY : CMakeFiles/gpd_hand.dir/build

CMakeFiles/gpd_hand.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gpd_hand.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gpd_hand.dir/clean

CMakeFiles/gpd_hand.dir/depend:
	cd /home/wenchang/projects/ForceRL/gpd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wenchang/projects/ForceRL/gpd /home/wenchang/projects/ForceRL/gpd /home/wenchang/projects/ForceRL/gpd/build /home/wenchang/projects/ForceRL/gpd/build /home/wenchang/projects/ForceRL/gpd/build/CMakeFiles/gpd_hand.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gpd_hand.dir/depend

