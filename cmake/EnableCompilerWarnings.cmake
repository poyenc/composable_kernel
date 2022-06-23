# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

# - Enable warning all for gcc/clang or use /W4 for visual studio

## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
    # set(CMAKE_CXX_WARNING_LEVEL 4)
    # if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # else ()
    #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    # endif ()

    # set(CMAKE_C_WARNING_LEVEL 4)
    # if (CMAKE_C_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    # else ()
    #     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # endif ()

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        # use -Wall for gcc and clang
        list(APPEND CMAKE_COMPILER_WARNINGS
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            # Shadow is broken on gcc when using lambdas
            # -Wshadow
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused

            -Wsign-compare
            -Wno-extra-semi-stmt
        )
        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Weverything
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-conversion
                -Wno-double-promotion
                -Wno-exit-time-destructors
                -Wno-extra-semi
                -Wno-float-conversion
                -Wno-gnu-anonymous-struct
                -Wno-gnu-zero-variadic-macro-arguments
                -Wno-missing-prototypes
                -Wno-nested-anon-types
                -Wno-padded
                -Wno-return-std-move-in-c++11
                -Wno-shorten-64-to-32
                -Wno-sign-conversion
                -Wno-unknown-warning-option
                -Wno-unused-command-line-argument
                -Wno-weak-vtables
                -Wno-covered-switch-default
            )
        else()
            if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "GNU" AND ${COMPILER} MATCHES "CXX")
                # cmake 3.5.2 does not support >=.
                if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.1")
                    list(APPEND CMAKE_COMPILER_WARNINGS
                        -Wno-ignored-attributes)
                endif()
            endif()
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wno-missing-field-initializers
                -Wno-deprecated-declarations
            )
        endif()
        add_definitions(${CMAKE_COMPILER_WARNINGS})
    endforeach()
endif ()
