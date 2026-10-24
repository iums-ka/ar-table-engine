
cmake_minimum_required(VERSION 3.20)

project(aruco_nano_ext LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(aruco_nano_ext python_bindings.cpp)

target_compile_features(aruco_nano_ext PRIVATE cxx_std_17)

target_include_directories(aruco_nano_ext PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(aruco_nano_ext PRIVATE
    ${OpenCV_LIBS}
)

# Put the generated extension directly into the Python package.
get_filename_component(
    PYTHON_PACKAGE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/ar_table_engine/vision/aruco"
    ABSOLUTE
)

set_target_properties(aruco_nano_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}"

    # Avoid a Release/ subdirectory with Visual Studio generators.
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PYTHON_PACKAGE_DIR}"
)
