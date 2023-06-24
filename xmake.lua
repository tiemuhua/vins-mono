add_rules("mode.debug", "mode.release")

set_languages("cxx17", "c17")

add_includedirs(
        "~/3rd/eigen/install_mac_m1_dir/include/eigen3",
        "~/3rd/ceres-solver/install_mac_m1_dir/include",
        "~/3rd/boost/install_mac_m1_dir/include",
        "~/3rd/DBoW2/install_mac_m1_dir/include",
        "~/3rd/opencv_and_contrib/install_mac_m1_dir/include/opencv4",
        "~/3rd/camodocal/include"
)

add_includedirs(
        "$(projectdir)/src/",
        "$(projectdir)/src/vins"
)

add_defines("CERES_EXPORT= ")

target("vins-mac")
    set_kind("static")
    add_files("$(projectdir)/src/vins/**.cpp")

--target("vins-ios")
--    add_rules("xcode.application")
--    set_kind("static")
--    add_files("$(projectdir)/src/vins/**.cpp")

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

