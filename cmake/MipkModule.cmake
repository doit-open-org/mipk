include_guard(GLOBAL)

# Module support for the mipk build.
#
# A module is a directory holding module.json and a CMakeLists.txt that calls
# mipk_add_module().  module.json names the module, the function that
# registers it, and the capabilities it provides; the same file feeds the
# packaging tool, so the runtime and the tool never disagree about what a
# module offers.
#
# Every enabled module ends up in the MIPK_ENABLED_MODULES global property.
# mipk_generate_registry() turns that list into a C file whose
# mipk_register_builtin_modules() registers each one, and
# mipk_write_capabilities() writes the capability catalog of a final target
# for the packaging tool to check manifests against.

define_property(GLOBAL PROPERTY MIPK_ENABLED_MODULES BRIEF_DOCS "Enabled module targets" FULL_DOCS "Enabled module targets")
define_property(GLOBAL PROPERTY MIPK_RUNTIME_DLLS BRIEF_DOCS "DLLs executables load" FULL_DOCS "Shared libraries (Windows DLLs) that backends need beside the executable")

# mipk_copy_runtime_dlls(<target>)
#
# Windows has no rpath: copies the DLLs enabled backends registered in
# MIPK_RUNTIME_DLLS next to <target> after it builds.  A no-op elsewhere.
function(mipk_copy_runtime_dlls target)
    get_property(_dlls GLOBAL PROPERTY MIPK_RUNTIME_DLLS)
    if(WIN32 AND _dlls)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls} "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()

function(_mipk_read_module_json path out_name out_register out_entries)
    file(READ "${path}" _json)
    string(JSON _name GET "${_json}" module)
    string(JSON _register GET "${_json}" register)
    string(JSON _count LENGTH "${_json}" capabilities)
    set(_entries)
    math(EXPR _last "${_count} - 1")
    foreach(_index RANGE ${_last})
        string(JSON _kind GET "${_json}" capabilities ${_index} kind)
        string(JSON _cap_name GET "${_json}" capabilities ${_index} name)
        string(JSON _abi GET "${_json}" capabilities ${_index} abi_version)
        string(JSON _version GET "${_json}" capabilities ${_index} implementation_version)
        list(APPEND _entries "${_kind}|${_cap_name}|${_abi}|${_version}")
    endforeach()
    set(${out_name} "${_name}" PARENT_SCOPE)
    set(${out_register} "${_register}" PARENT_SCOPE)
    set(${out_entries} "${_entries}" PARENT_SCOPE)
endfunction()

# mipk_add_module(SOURCES ... [INCLUDE_DIRS ...] [LINK ...] [DEFINITIONS ...])
#
# Creates an object library named mipk_<module> from the module.json next to
# the calling CMakeLists.txt and records it as enabled.  The objects of every
# enabled module are archived into libmipk.a together with core, so the
# mutual references between core's generated registry and the modules resolve
# inside one archive.
function(mipk_add_module)
    cmake_parse_arguments(MOD "" "" "SOURCES;INCLUDE_DIRS;LINK;DEFINITIONS;COMPILE_OPTIONS" ${ARGN})
    set(_json "${CMAKE_CURRENT_SOURCE_DIR}/module.json")
    if(NOT EXISTS "${_json}")
        message(FATAL_ERROR "mipk_add_module: ${CMAKE_CURRENT_SOURCE_DIR} has no module.json")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_json}")
    _mipk_read_module_json("${_json}" _name _register _entries)
    set(_target "mipk_${_name}")
    add_library(${_target} OBJECT ${MOD_SOURCES})
    add_library(mipk::${_name} ALIAS ${_target})
    target_link_libraries(${_target} PUBLIC mipk::core_headers ${MOD_LINK})
    target_include_directories(${_target}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${MOD_INCLUDE_DIRS})
    target_compile_definitions(${_target} PRIVATE ${MOD_DEFINITIONS})
    target_compile_options(${_target} PRIVATE ${MIPK_WARNING_FLAGS} ${MOD_COMPILE_OPTIONS})
    set_target_properties(${_target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        MIPK_MODULE_NAME "${_name}"
        MIPK_MODULE_REGISTER "${_register}"
        MIPK_CAPABILITIES "${_entries}"
        MIPK_MODULE_JSON "${_json}")
    set_property(GLOBAL APPEND PROPERTY MIPK_ENABLED_MODULES ${_target})
endfunction()

# mipk_generate_registry(OUTPUT <file>)
#
# Writes the C file that registers every enabled module.
function(mipk_generate_registry)
    cmake_parse_arguments(GEN "" "OUTPUT" "" ${ARGN})
    get_property(_modules GLOBAL PROPERTY MIPK_ENABLED_MODULES)
    set(_declarations "")
    set(_calls "")
    foreach(_module IN LISTS _modules)
        get_target_property(_register ${_module} MIPK_MODULE_REGISTER)
        get_target_property(_name ${_module} MIPK_MODULE_NAME)
        string(APPEND _declarations "mipk_status ${_register}(void);\n")
        string(APPEND _calls "    status = ${_register}();\n    if (status != MIPK_OK) {\n        fprintf(stderr, \"mipk: module ${_name} failed to register (%d)\\n\", (int)status);\n        return status;\n    }\n")
    endforeach()
    get_filename_component(_gen_name "${GEN_OUTPUT}" NAME)
    set(_gen_staging "${CMAKE_BINARY_DIR}/mipk_staging/${_gen_name}.in")
    file(WRITE "${_gen_staging}"
"/* Generated by mipk_generate_registry(); do not edit. */
#include \"mipk_module.h\"

#include <stdio.h>

${_declarations}
mipk_status mipk_register_builtin_modules(void)
{
    mipk_status status = MIPK_OK;
${_calls}    return status;
}
")
    configure_file("${_gen_staging}" "${GEN_OUTPUT}" COPYONLY)
endfunction()

function(_mipk_collect_capabilities _target _visited_var _entries_var)
    set(_visited ${${_visited_var}})
    set(_entries ${${_entries_var}})
    if(NOT TARGET "${_target}")
        set(${_visited_var} "${_visited}" PARENT_SCOPE)
        set(${_entries_var} "${_entries}" PARENT_SCOPE)
        return()
    endif()
    get_target_property(_aliased "${_target}" ALIASED_TARGET)
    if(_aliased)
        set(_target "${_aliased}")
    endif()
    if("${_target}" IN_LIST _visited)
        set(${_visited_var} "${_visited}" PARENT_SCOPE)
        set(${_entries_var} "${_entries}" PARENT_SCOPE)
        return()
    endif()
    list(APPEND _visited "${_target}")
    get_target_property(_own "${_target}" MIPK_CAPABILITIES)
    if(_own)
        list(APPEND _entries ${_own})
    endif()
    get_target_property(_type "${_target}" TYPE)
    set(_links)
    if(NOT _type STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_direct "${_target}" LINK_LIBRARIES)
        if(_direct)
            list(APPEND _links ${_direct})
        endif()
    endif()
    get_target_property(_interface "${_target}" INTERFACE_LINK_LIBRARIES)
    if(_interface)
        list(APPEND _links ${_interface})
    endif()
    foreach(_link IN LISTS _links)
        string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" _link "${_link}")
        _mipk_collect_capabilities("${_link}" _visited _entries)
    endforeach()
    set(${_visited_var} "${_visited}" PARENT_SCOPE)
    set(${_entries_var} "${_entries}" PARENT_SCOPE)
endfunction()

# mipk_write_capabilities(TARGET <exe> OUTPUT <json>)
#
# Writes the capabilities reachable from a target's link closure, in the
# format mipk_tool.py --target-capabilities expects.
function(mipk_write_capabilities)
    cmake_parse_arguments(CAP "" "TARGET;OUTPUT" "" ${ARGN})
    set(_visited)
    set(_entries)
    _mipk_collect_capabilities("${CAP_TARGET}" _visited _entries)
    list(REMOVE_DUPLICATES _entries)
    list(SORT _entries)
    set(_items "")
    foreach(_entry IN LISTS _entries)
        string(REPLACE "|" ";" _fields "${_entry}")
        list(GET _fields 0 _kind)
        list(GET _fields 1 _name)
        list(GET _fields 2 _abi)
        list(GET _fields 3 _version)
        if(_items)
            string(APPEND _items ",\n")
        endif()
        string(APPEND _items "    {\"kind\": \"${_kind}\", \"name\": \"${_name}\", \"abi_version\": ${_abi}, \"implementation_version\": \"${_version}\"}")
    endforeach()
    get_filename_component(_cap_name "${CAP_OUTPUT}" NAME)
    set(_cap_staging "${CMAKE_BINARY_DIR}/mipk_staging/${CAP_TARGET}_${_cap_name}.in")
    file(WRITE "${_cap_staging}" "{\n  \"capabilities\": [\n${_items}\n  ]\n}\n")
    configure_file("${_cap_staging}" "${CAP_OUTPUT}" COPYONLY)
endfunction()
