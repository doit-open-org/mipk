include(HiSDK)
mipk_require_hisdk()
mipk_add_module(SOURCES mipk_svp_acl.c LINK mipk::hisdk)
