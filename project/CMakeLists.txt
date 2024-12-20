cmake_minimum_required(VERSION 3.10)

project(openclnn_project
    VERSION 1.0
    DESCRIPTION "Semantic Segmentation Project Suite"
    LANGUAGES CXX
)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories for all targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable folders for IDE organization
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add the subdirectories in the correct order
add_subdirectory(openclnn)
add_subdirectory(app)