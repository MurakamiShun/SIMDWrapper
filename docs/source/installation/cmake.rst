#######################
Installation with CMake
#######################


Installation
============

We can use CMake to install SIMDWrapper package.

.. code-block:: bash

    $ git clone https://github.com/akisute514/SIMDWrapper.git
    $ cd SIMDWrapper
    $ mkdir build && cd build
    $ cmake .. && make install


Quick Start
===========

Make your project folder and source files.
And make CMakeLists.txt to build our source under under your project folder.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.9)

    project(your_prject_name LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    find_package(SIMDWrapper)
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE SIMDWrapper)

Make build folder and type commands to build.

.. code-block:: bash

    $ cmake ..
    $ make

The project folder become like this. And we can find executable file in build folder. 

.. code-block::

    +- main.cpp
    +- build
    |   +- your_prject_name (executable file)
    |   +- etc...
    +- CMakeLists.txt

    