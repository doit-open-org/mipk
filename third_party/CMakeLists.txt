add_library(mipk_cjson STATIC cjson/cJSON.c)
add_library(mipk::cjson ALIAS mipk_cjson)
target_include_directories(mipk_cjson PUBLIC cjson)
set_target_properties(mipk_cjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MIPK_OPS_AUDIO)
    file(GLOB _microfrontend_sources
        tensorflow/lite/experimental/microfrontend/lib/*.cc)
    add_library(mipk_microfrontend STATIC ${_microfrontend_sources})
    add_library(mipk::microfrontend ALIAS mipk_microfrontend)
    target_include_directories(mipk_microfrontend PUBLIC . kissfft)
    target_compile_options(mipk_microfrontend PRIVATE -Wno-unused-parameter)
    set_target_properties(mipk_microfrontend PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
