cmake_minimum_required(VERSION 3.10)

# Enforce out-of-source builds
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

# Project name
project(face_detect_app
    VERSION 1.0
    DESCRIPTION "Semantic Segmentation Application"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable IDE folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Source files
set(SOURCE_FILES
    src/bmpreader.cpp
    main.cpp
)

# Header files
set(HEADER_FILES
    include/bmpreader.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# User provides only the root directory for openclnn
set(OPENCLNN_ROOT_DIR "" CACHE PATH "Root directory for openclnn (e.g., C:/Program Files (x86)/openclnn)")

# Ensure the openclnn root directory is provided
if(NOT OPENCLNN_ROOT_DIR)
    message(FATAL_ERROR "Please provide the OPENCLNN_ROOT_DIR path (e.g., C:/Program Files (x86)/openclnn).")
endif()

# Construct include and lib directories relative to the provided root directory
set(OPENCLNN_INCLUDE_DIR "${OPENCLNN_ROOT_DIR}/include")
set(OPENCLNN_LIB_DIR "${OPENCLNN_ROOT_DIR}/lib")

# Check if the openclnn directories exist
if(NOT EXISTS ${OPENCLNN_INCLUDE_DIR})
    message(FATAL_ERROR "The provided OPENCLNN_INCLUDE_DIR does not exist: ${OPENCLNN_INCLUDE_DIR}")
endif()

if(NOT EXISTS ${OPENCLNN_LIB_DIR})
    message(FATAL_ERROR "The provided OPENCLNN_LIB_DIR does not exist: ${OPENCLNN_LIB_DIR}")
endif()

# Add openclnn include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${OPENCLNN_INCLUDE_DIR})

# Find openclnn library
if(WIN32)
    set(OPENCLNN_LIB_NAME "openclnn.lib")
else()
    set(OPENCLNN_LIB_NAME "libopenclnn.so")
endif()

# Ensure the library path is correct
find_library(OPENCLNN_LIBRARY
    NAMES ${OPENCLNN_LIB_NAME}
    PATHS ${OPENCLNN_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

# Link against openclnn
target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENCLNN_LIBRARY})

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/W4 /WX- /MP /Zi /Od /RTC1>
        $<$<CONFIG:Release>:/W4 /WX- /MP /O2 /Oi /GL>
    )
    # Add Unicode support for Windows
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        UNICODE 
        _UNICODE
        WIN32
        _CONSOLE
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O3>
    )
endif()

# Installation rules
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
