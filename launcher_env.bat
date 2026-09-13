@echo off
set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "SERVE_BIN=%ROOT%build\apps\ninfer-serve.exe"
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"

if not exist "%MODEL%" (
    if exist "%ROOT%models\qwen3_8_27b_nvfp4qat.ninfer" (
        set "MODEL=%ROOT%models\qwen3_8_27b_nvfp4qat.ninfer"
    )
)

:: Locate suitable Python executable
set "PYTHON_EXE="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=py"
    )
)
