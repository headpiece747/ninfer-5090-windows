@echo off
setlocal enabledelayedexpansion
echo ========================================================
echo NInfer Windows Release Packager
echo ========================================================

set VERSION=%1
if "%VERSION%"=="" (
    set VERSION=v1.0.5
)

echo Packaging version: %VERSION%

if not exist build\apps\ninfer-serve.exe (
    echo Error: build\apps\ninfer-serve.exe not found! Please build first.
    exit /b 1
)

set TEXT_DIR=staging_release_text
set VISION_DIR=staging_release_vision

if exist %TEXT_DIR% rmdir /S /Q %TEXT_DIR%
if exist %VISION_DIR% rmdir /S /Q %VISION_DIR%

mkdir %TEXT_DIR%
mkdir %VISION_DIR%

echo Copying common files...
copy NOTICE %TEXT_DIR%\ >nul
copy LICENSE %TEXT_DIR%\ >nul
copy README.md %TEXT_DIR%\ >nul
copy RELEASE_NOTES.md %TEXT_DIR%\ >nul
copy download_model.bat %TEXT_DIR%\ >nul
copy start_ninfer_5090.bat %TEXT_DIR%\ >nul
copy start_ninfer_dflash2.bat %TEXT_DIR%\ >nul
copy build\apps\ninfer-serve.exe %TEXT_DIR%\ >nul
copy build\apps\ninfer.exe %TEXT_DIR%\ >nul
copy build\apps\ninfer-perplexity.exe %TEXT_DIR%\ >nul

copy NOTICE %VISION_DIR%\ >nul
copy LICENSE %VISION_DIR%\ >nul
copy README.md %VISION_DIR%\ >nul
copy RELEASE_NOTES.md %VISION_DIR%\ >nul
copy download_model.bat %VISION_DIR%\ >nul
copy start_ninfer_vision.bat %VISION_DIR%\ >nul
copy build\apps\ninfer-serve.exe %VISION_DIR%\ >nul
copy build\apps\ninfer.exe %VISION_DIR%\ >nul
copy build\apps\ninfer-perplexity.exe %VISION_DIR%\ >nul

if exist ffmpeg\bin (
    echo Copying FFmpeg DLLs...
    copy ffmpeg\bin\avcodec-*.dll %VISION_DIR%\ >nul 2>nul
    copy ffmpeg\bin\avformat-*.dll %VISION_DIR%\ >nul 2>nul
    copy ffmpeg\bin\avutil-*.dll %VISION_DIR%\ >nul 2>nul
    copy ffmpeg\bin\swscale-*.dll %VISION_DIR%\ >nul 2>nul
    copy ffmpeg\bin\swresample-*.dll %VISION_DIR%\ >nul 2>nul
)

echo Compressing archives...
powershell -Command "Compress-Archive -Path '%TEXT_DIR%\*' -DestinationPath 'ninfer-windows-%VERSION%-rtx5090.zip' -Force"
powershell -Command "Compress-Archive -Path '%VISION_DIR%\*' -DestinationPath 'ninfer-windows-vision-%VERSION%-rtx5090.zip' -Force"

echo Cleaning up staging directories...
rmdir /S /Q %TEXT_DIR%
rmdir /S /Q %VISION_DIR%

echo.
echo ========================================================
echo Packaging Complete!
echo Created: ninfer-windows-%VERSION%-rtx5090.zip
echo Created: ninfer-windows-vision-%VERSION%-rtx5090.zip
echo ========================================================
