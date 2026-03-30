@echo off
REM WhisperLiveKit Quick Install Script for Windows
REM Usage: scripts\install-wlk.bat

echo ==========================================
echo   WhisperLiveKit Quick Install
echo ==========================================
echo.

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ uv is not installed. Installing uv...
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo Please restart your terminal and run this script again.
    exit /b 1
)

echo ✅ uv is installed
echo.

REM Install WhisperLiveKit
echo 📦 Installing WhisperLiveKit...
uv pip install whisperlivekit

REM Install audio capture dependencies
echo.
echo 📦 Installing audio capture dependencies...
echo Installing sounddevice (cross-platform)...
uv pip install sounddevice

echo.
echo Installing pyaudio (optional, requires portaudio)...
uv pip install pyaudio 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ pyaudio installed successfully
) else (
    echo ⚠️  pyaudio installation failed (this is OK, sounddevice will be used)
)

REM Ask about diarization support
echo.

REM Ask about diarization support
echo.
echo ==========================================
echo   Diarization Support (Speaker ID)
echo ==========================================
echo.
echo Do you want to install diarization support?
echo This requires NVIDIA NeMo (large download ~1GB)
echo.
set /p INSTALL_DIARIZATION="Install diarization? (y/N): "
if /i "%INSTALL_DIARIZATION%"=="y" (
    echo.
    echo 📦 Installing NeMo for diarization...
    uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
    echo ✅ Diarization support installed
) else (
    echo.
    echo ⚠️  Skipping diarization. You can still use WLK without --diarization flag
)

REM Verify installation
echo.
echo 🔍 Verifying installation...
uv run wlk --help >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo   Installation Complete!
    echo ==========================================
    echo.
    echo Quick start:
    echo   # Start server (without diarization)
    echo   uv run wlk --model base --language zh
    echo.
    echo   # Start server (with diarization, if installed)
    echo   uv run wlk --model base --language zh --diarization
    echo.
    echo   # Test with zoom-ai
    echo   uv run python -m zoom_ai.cli test-wlk --duration 60
) else (
    echo ❌ Installation verification failed
    exit /b 1
)

echo.
pause
