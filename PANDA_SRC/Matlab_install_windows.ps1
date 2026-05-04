Write-Host "Creating Visual Studio build system in ./build folder"
Write-Host "Make sure you have Visual Studio with C/C++ tools and CMake installed"

cmake . -Bbuild -DCMAKE_GENERATOR_PLATFORM=x64

Write-Host "Compiling 64 bit static library used by the mex file"
cmake --build ./build --config Release

cp .\build\PANDA\Release\PANDA_lib.lib .\Matlab\PANDA_lib.lib

Write-Host "Compiling the mex file"
mex .\Matlab\panda.c .\Matlab\arguments_check.c .\Matlab\arguments_parse.c .\Matlab\PANDA_lib.lib -outdir .\Matlab