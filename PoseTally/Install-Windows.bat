@echo off
rem PoseTally installer for Windows. Double-click this file.
rem (If Windows shows "Windows protected your PC": click "More info", then "Run anyway".)
setlocal
title PoseTally installer

if not exist "%~dp0installer\install_windows.ps1" goto :not_unpacked
if not exist "%~dp0app\posetally\__init__.py" goto :not_unpacked

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install_windows.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo The installation did not finish.
    echo Read the messages above. When the problem is fixed, run this file again:
    echo it continues where it stopped.
)
if not defined POSETALLY_NO_PAUSE pause
exit /b %CODE%

:not_unpacked
echo The PoseTally program files were not found next to this installer.
echo.
echo Unpack the whole ZIP file first: right-click the ZIP file, choose "Extract All...",
echo then open the unpacked folder and run Install-Windows.bat from there.
echo.
if not defined POSETALLY_NO_PAUSE pause
exit /b 1
