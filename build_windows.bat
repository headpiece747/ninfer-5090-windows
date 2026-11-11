@echo off
echo ========================================================
echo NInfer Build Script (Windows MSVC)
echo ========================================================

REM FFmpeg's LGPL shared build, not the GPL one: these DLLs ship inside the release archive, and
REM the LGPL variant keeps GPL components out of it. BtbN prunes old autobuilds, so the pinned tag
REM is tried first with `latest` as the fallback -- this pin went stale once already.
set "FFMPEG_ASSET=ffmpeg-master-latest-win64-lgpl-shared.zip"
set "FFMPEG_PIN=autobuild-2026-09-20-13-11"
if not exist ffmpeg (
    echo [1/4] Downloading FFmpeg Windows dev binaries ^(LGPL shared^)...
    powershell -Command "try { Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/%FFMPEG_PIN%/%FFMPEG_ASSET%' -OutFile 'ffmpeg.zip' } catch { Write-Host 'Pinned autobuild is gone; falling back to the latest release.'; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/%FFMPEG_ASSET%' -OutFile 'ffmpeg.zip' }"
    
    echo [2/4] Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_temp' -Force"
    
    echo Moving files into place...
    move ffmpeg_temp\ffmpeg-master-latest-win64-lgpl-shared ffmpeg >nul
    
    echo Cleaning up...
    rmdir /S /Q ffmpeg_temp
    del ffmpeg.zip
) else (
    echo [1/4] FFmpeg directory already exists. Skipping download.
)

REM libjpeg-turbo: a self-extracting MSVC installer (not a zip). Stage it the same way FFmpeg is
REM staged -- header + import lib + DLL -- so the JPEG::JPEG target in Dependencies.cmake resolves.
set "JPEG_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0-vc-x64.exe"
if not exist jpeg (
    echo Downloading libjpeg-turbo Windows dev binaries...
    powershell -Command "Invoke-WebRequest -Uri '%JPEG_URL%' -OutFile 'libjpeg-turbo.exe'"
    echo Extracting libjpeg-turbo...
    libjpeg-turbo.exe /extract=jpeg_temp /nosplash /silent
    echo Moving files into place...
    mkdir jpeg\include 2>nul & mkdir jpeg\lib 2>nul & mkdir jpeg\bin 2>nul
    xcopy /y /q "jpeg_temp\include" "jpeg\include\" >nul
    copy /y "jpeg_temp\lib\jpeg*.lib" "jpeg\lib\" >nul
    copy /y "jpeg_temp\bin\jpeg*.dll" "jpeg\bin\" >nul
    echo Cleaning up...
    rmdir /S /Q jpeg_temp
    del libjpeg-turbo.exe
) else (
    echo libjpeg-turbo directory already exists. Skipping download.
)

echo.
echo [3/4] Setting up MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul || echo Please open x64 Native Tools Command Prompt manually!
REM The chain above ends with an echo, so its errorlevel cannot report whether a vcvars was found.
REM Ask for cmake directly instead: without this check the script continues and fails somewhere in
REM the configure step, which reads as a project problem rather than a missing environment.
where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No cmake on PATH. Install the Visual Studio C++ workload, or run this from an x64
    echo         Native Tools Command Prompt.
    exit /b 1
)

echo.
echo [4/4] Compiling NInfer with Vision Support...
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="120a" -DCMAKE_BUILD_TYPE=Release
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b 1

cmake --build build --config Release -j
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b 1

echo.
echo Copying FFmpeg DLLs to the build folder so the executable can find them...
for %%D in (avcodec avformat avutil swscale swresample) do xcopy /y /q "ffmpeg\bin\%%D-*.dll" "build\apps\" >nul
copy /y "jpeg\bin\jpeg*.dll" "build\apps\" >nul

echo.
echo ========================================================
echo Build Complete!
echo Executable is located at: build\apps\ninfer-serve.exe
echo ========================================================
pause
