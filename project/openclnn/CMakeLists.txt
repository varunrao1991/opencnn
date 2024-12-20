cmake_minimum_required(VERSION 3.10)

# Enforce building in a separate directory
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

project(openclnn 
    VERSION 1.0
    DESCRIPTION "Semantic Segmentation Library with OpenCL Support"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define OpenCL paths
set(OPENCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/opencl")
set(OPENCL_INCLUDE_DIR "${OPENCL_ROOT}/include")
set(OPENCL_LIB_DIR "${OPENCL_ROOT}/lib/x64")

# Set output directories relative to build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Define source files
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Define header files - split into public and private
file(GLOB OPENCLNN_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/openclnn/*.h")
file(GLOB PRIVATE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

# Get kernel files
file(GLOB KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*")

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${OPENCLNN_PUBLIC_HEADERS} ${PRIVATE_HEADERS})

# Include directories for compilation
target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${OPENCL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG ENABLE_INC_TIMER_GPUML SIMPLE_DUMP>
    MYDLL_EXPORTS
)

# Platform specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32
        _CONSOLE
        UNICODE
        _UNICODE
    )
    
    # Link OpenCL libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OPENCL_LIB_DIR}/OpenCL.lib
        $<$<CONFIG:Debug>:${OPENCL_LIB_DIR}/oclUtils64D.lib>
        $<$<CONFIG:Release>:${OPENCL_LIB_DIR}/oclUtils64.lib>
    )
else()
    # Unix systems typically have OpenCL in standard locations
    find_package(OpenCL REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCL_LIBRARIES}
        dl
        pthread
    )
endif()

# Compiler options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra $<$<CONFIG:Release>:-O3>)
endif()

# Installation rules
include(GNUInstallDirs)

# Install library
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install only the public headers (from openclnn folder)
install(FILES ${OPENCLNN_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openclnn
)

# Install kernel files
install(FILES ${KERNEL_FILES}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/kernels
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Add message to verify paths
message(STATUS "OpenCL Include Dir: ${OPENCL_INCLUDE_DIR}")
message(STATUS "OpenCL Library Dir: ${OPENCL_LIB_DIR}")
message(STATUS "Public Headers: ${OPENCLNN_PUBLIC_HEADERS}")
message(STATUS "Kernel Files: ${KERNEL_FILES}")