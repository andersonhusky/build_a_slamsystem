# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/hongfeng/CV/build_a_slam_bymyself/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hongfeng/CV/build_a_slam_bymyself/build

# Include any dependencies generated for this target.
include slam01/CMakeFiles/Construct2.dir/depend.make

# Include the progress variables for this target.
include slam01/CMakeFiles/Construct2.dir/progress.make

# Include the compile flags for this target's objects.
include slam01/CMakeFiles/Construct2.dir/flags.make

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o: slam01/CMakeFiles/Construct2.dir/flags.make
slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o: /home/hongfeng/CV/build_a_slam_bymyself/src/slam01/src/Construct2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hongfeng/CV/build_a_slam_bymyself/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o"
	cd /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Construct2.dir/src/Construct2.cpp.o -c /home/hongfeng/CV/build_a_slam_bymyself/src/slam01/src/Construct2.cpp

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Construct2.dir/src/Construct2.cpp.i"
	cd /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hongfeng/CV/build_a_slam_bymyself/src/slam01/src/Construct2.cpp > CMakeFiles/Construct2.dir/src/Construct2.cpp.i

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Construct2.dir/src/Construct2.cpp.s"
	cd /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hongfeng/CV/build_a_slam_bymyself/src/slam01/src/Construct2.cpp -o CMakeFiles/Construct2.dir/src/Construct2.cpp.s

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.requires:

.PHONY : slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.requires

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.provides: slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.requires
	$(MAKE) -f slam01/CMakeFiles/Construct2.dir/build.make slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.provides.build
.PHONY : slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.provides

slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.provides.build: slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o


# Object files for target Construct2
Construct2_OBJECTS = \
"CMakeFiles/Construct2.dir/src/Construct2.cpp.o"

# External object files for target Construct2
Construct2_EXTERNAL_OBJECTS =

/home/hongfeng/CV/build_a_slam_bymyself/devel/lib/libConstruct2.so: slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o
/home/hongfeng/CV/build_a_slam_bymyself/devel/lib/libConstruct2.so: slam01/CMakeFiles/Construct2.dir/build.make
/home/hongfeng/CV/build_a_slam_bymyself/devel/lib/libConstruct2.so: slam01/CMakeFiles/Construct2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hongfeng/CV/build_a_slam_bymyself/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library /home/hongfeng/CV/build_a_slam_bymyself/devel/lib/libConstruct2.so"
	cd /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Construct2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
slam01/CMakeFiles/Construct2.dir/build: /home/hongfeng/CV/build_a_slam_bymyself/devel/lib/libConstruct2.so

.PHONY : slam01/CMakeFiles/Construct2.dir/build

slam01/CMakeFiles/Construct2.dir/requires: slam01/CMakeFiles/Construct2.dir/src/Construct2.cpp.o.requires

.PHONY : slam01/CMakeFiles/Construct2.dir/requires

slam01/CMakeFiles/Construct2.dir/clean:
	cd /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 && $(CMAKE_COMMAND) -P CMakeFiles/Construct2.dir/cmake_clean.cmake
.PHONY : slam01/CMakeFiles/Construct2.dir/clean

slam01/CMakeFiles/Construct2.dir/depend:
	cd /home/hongfeng/CV/build_a_slam_bymyself/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hongfeng/CV/build_a_slam_bymyself/src /home/hongfeng/CV/build_a_slam_bymyself/src/slam01 /home/hongfeng/CV/build_a_slam_bymyself/build /home/hongfeng/CV/build_a_slam_bymyself/build/slam01 /home/hongfeng/CV/build_a_slam_bymyself/build/slam01/CMakeFiles/Construct2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : slam01/CMakeFiles/Construct2.dir/depend
