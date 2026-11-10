@echo off
rem Removes PoseTally from this computer. Double-click this file.
rem Your DeepLabCut projects, videos, results and saved scoring setups are NOT touched.
setlocal
title Remove PoseTally

if defined POSETALLY_HOME (set "APPHOME=%POSETALLY_HOME%") else (set "APPHOME=%LOCALAPPDATA%\PoseTally")

echo This removes PoseTally and its private copy of Python and DeepLabCut:
echo   %APPHOME%
echo Your projects, videos, results and saved scoring setups are not touched.
echo.
if defined POSETALLY_NO_PAUSE goto :remove
set /p ANSWER=Type yes and press Enter to remove it: 
if /i not "%ANSWER%"=="yes" (
    echo Nothing was removed.
    pause
    exit /b 0
)

:remove
if "%APPHOME%"=="" goto :refuse
if /i "%APPHOME%"=="%USERPROFILE%" goto :refuse
if exist "%APPHOME%" rmdir /s /q "%APPHOME%"
powershell -NoProfile -Command "$d=[Environment]::GetFolderPath('Desktop'); Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $d 'PoseTally.lnk'); Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\PoseTally.lnk')"
echo PoseTally was removed.
if not defined POSETALLY_NO_PAUSE pause
exit /b 0

:refuse
echo Refusing to remove "%APPHOME%".
if not defined POSETALLY_NO_PAUSE pause
exit /b 1
