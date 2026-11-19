@echo off
REM Double-click this file. Needs MSYS2 installed at C:\msys64 (https://www.msys2.org)
cd /d "%~dp0"
if not exist C:\msys64\usr\bin\bash.exe (
  echo MSYS2 not found. Install it from https://www.msys2.org to C:\msys64 and run this again.
  pause
  exit /b 1
)
set MSYSTEM=MINGW64
set CHERE_INVOKING=1
C:\msys64\usr\bin\bash.exe -lc "./build_windows.sh"
if exist car_studio.exe (
  echo.
  echo SUCCESS - starting car_studio.exe
  start "" car_studio.exe
) else (
  echo.
  echo Build failed. Copy the messages above and send them to me.
)
pause
