# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.


if(NOT TARGET analyze)
    add_custom_target(analyze)
endif()

function(mark_as_analyzer)
    add_dependencies(analyze ${ARGN})
endfunction()

