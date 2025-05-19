cmake_minimum_required(VERSION 3.10)
project(lr2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCES
    main.cpp
    impl/zero.cpp
    impl/first.cpp
    impl/second.cpp
    impl/thirt.cpp
    impl/fourth.cpp
    impl/fiveth.cpp
    impl/sixth.cpp

)

add_executable(lr2 ${SOURCES})

target_include_directories(lr2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_compile_options(lr2 PUBLIC
    -Wall
    -Wextra
    -pedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(lr2 PUBLIC -O3)
endif()