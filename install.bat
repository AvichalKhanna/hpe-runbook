@echo off
setlocal enabledelayedexpansion
title Stratum AI - Installation

echo ========================================================
echo   Stratum AI - Runbook RAG Chatbot Installation
echo ========================================================
echo.

:: 1. Fix working directory
cd /d "%~dp0"

:: 2. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 goto :NO_PYTHON
goto :PYTHON_OK

:NO_PYTHON
echo [ERROR] Python is not installed or not in PATH.
echo Please install Python 3.9 or higher and check "Add to PATH".
goto :FAIL

:PYTHON_OK
:: 3. Check Git (Informational warning)
git --version >nul 2>&1
if %errorlevel% equ 0 goto :GIT_OK
echo [WARNING] Git is not installed. Some pip dependencies might fail if they require compiling from source.
:GIT_OK

:: 4. Create venv
if exist "venv\" goto :VENV_EXISTS
echo [INFO] Creating Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 goto :VENV_FAIL
goto :VENV_CREATED

:VENV_FAIL
echo [ERROR] Failed to create virtual environment.
goto :FAIL

:VENV_EXISTS
echo [INFO] Virtual environment already exists.
:VENV_CREATED

:: 5. Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 goto :ACTIVATE_FAIL
goto :ACTIVATE_OK

:ACTIVATE_FAIL
echo [ERROR] Failed to activate virtual environment.
goto :FAIL

:ACTIVATE_OK
:: 6. Upgrade pip and install build tools
echo [INFO] Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 goto :PIP_FAIL

:: 6a. Install llama-cpp-python (CPU-only pre-built binary, no C++ compiler needed)
:: --only-binary :all: forces pip to NEVER build from source. If no pre-built wheel
:: exists for the current Python version, it will try the fallback index instead.
echo [INFO] Installing llama-cpp-python (CPU binary)...
pip install llama-cpp-python --only-binary :all: --prefer-binary --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Standard binary not found. Trying fallback CPU wheel index...
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all: --prefer-binary --quiet
)
if %errorlevel% neq 0 (
    echo [ERROR] Could not install llama-cpp-python.
    echo [INFO]  Try installing Visual Studio Build Tools from:
    echo [INFO]  https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo [INFO]  Then re-run this installer.
    goto :FAIL
)
echo [OK] llama-cpp-python installed.

:: 6b. Install all other requirements (excluding llama-cpp-python to avoid recompile)
echo [INFO] Installing remaining requirements...
pip install -r requirements.txt --prefer-binary --ignore-installed llama-cpp-python --quiet
if %errorlevel% neq 0 goto :PIP_FAIL
goto :PIP_OK

:PIP_FAIL
echo [ERROR] Failed to install requirements.
goto :FAIL

:PIP_OK
:: 7. Create required directories
echo [INFO] Creating required directories...
mkdir uploads 2>nul
mkdir index_store 2>nul
mkdir cache 2>nul
mkdir logs 2>nul
mkdir temp 2>nul
mkdir models 2>nul
mkdir exports 2>nul

:: 8. Initialize empty chunks.json if absent
if exist "index_store\chunks.json" goto :CHUNKS_OK
echo [] > "index_store\chunks.json"
:CHUNKS_OK

:: 9. Download models
echo [INFO] Downloading required models...
python download_models.py
if %errorlevel% neq 0 goto :MODELS_FAIL
goto :MODELS_OK

:MODELS_FAIL
echo [ERROR] Model download failed.
goto :FAIL

:MODELS_OK
:: 10. Post-install self-check
echo [INFO] Running post-installation self-checks...
python -c "import faiss, rank_bm25, fastapi, transformers, yaml, psutil, duckduckgo_search, trafilatura, websockets; print('[OK] Core libraries imported.')"
if %errorlevel% neq 0 goto :CHECKS_FAIL
goto :CHECKS_OK

:CHECKS_FAIL
echo [ERROR] Core libraries failed to import.
goto :FAIL

:CHECKS_OK
echo.
echo [INFO] Checking GPU / Hardware Acceleration support...
python check_gpu.py
echo.
echo ========================================================
echo   INSTALLATION COMPLETE - ALL SYSTEMS GO
echo ========================================================
echo.
echo You can now start the application by running START_APP.bat
echo.
pause
exit /b 0

:FAIL
echo.
echo ========================================================
echo   INSTALLATION FAILED
echo ========================================================
echo Please check the error messages above.
echo.
pause
exit /b 1
