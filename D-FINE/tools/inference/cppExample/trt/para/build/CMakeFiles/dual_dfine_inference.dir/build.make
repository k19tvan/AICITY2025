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
CMAKE_SOURCE_DIR = /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build

# Include any dependencies generated for this target.
include CMakeFiles/dual_dfine_inference.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dual_dfine_inference.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dual_dfine_inference.dir/flags.make

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o: CMakeFiles/dual_dfine_inference.dir/flags.make
CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o: ../dual_dfine_inference.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o -c /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/dual_dfine_inference.cpp

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/dual_dfine_inference.cpp > CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.i

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/dual_dfine_inference.cpp -o CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.s

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.requires:

.PHONY : CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.requires

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.provides: CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.requires
	$(MAKE) -f CMakeFiles/dual_dfine_inference.dir/build.make CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.provides.build
.PHONY : CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.provides

CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.provides.build: CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o


# Object files for target dual_dfine_inference
dual_dfine_inference_OBJECTS = \
"CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o"

# External object files for target dual_dfine_inference
dual_dfine_inference_EXTERNAL_OBJECTS =

dual_dfine_inference: CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o
dual_dfine_inference: CMakeFiles/dual_dfine_inference.dir/build.make
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libnvinfer.so.10
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10
dual_dfine_inference: /usr/local/cuda-11.8/lib64/libcudart_static.a
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/librt.so
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
dual_dfine_inference: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
dual_dfine_inference: CMakeFiles/dual_dfine_inference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dual_dfine_inference"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dual_dfine_inference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dual_dfine_inference.dir/build: dual_dfine_inference

.PHONY : CMakeFiles/dual_dfine_inference.dir/build

CMakeFiles/dual_dfine_inference.dir/requires: CMakeFiles/dual_dfine_inference.dir/dual_dfine_inference.cpp.o.requires

.PHONY : CMakeFiles/dual_dfine_inference.dir/requires

CMakeFiles/dual_dfine_inference.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dual_dfine_inference.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dual_dfine_inference.dir/clean

CMakeFiles/dual_dfine_inference.dir/depend:
	cd /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build /workspace/Repos/maibel/D-FINE/tools/inference/cppExample/trt/para/build/CMakeFiles/dual_dfine_inference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dual_dfine_inference.dir/depend

