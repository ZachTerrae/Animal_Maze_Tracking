# PoseTally installer for Windows (first stage).
#
# It downloads the "uv" package tool, installs a private copy of Python and creates
# an empty environment, all inside %LOCALAPPDATA%\PoseTally. The rest of the
# installation is done by installer\posetally_setup.py.
#
# Nothing outside that folder is changed, except for the shortcuts that start
# PoseTally. Running the installer again repairs or updates the installation.
#
# Keep this file plain ASCII: Windows PowerShell 5 misreads other characters.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$UvVersion = "0.13.7"
$PythonVersion = "3.12"

function Stop-Install([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    exit 1
}

Write-Host "============================================================"
Write-Host " PoseTally installer"
Write-Host "============================================================"

$Source = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Source "app\posetally\__init__.py")) -or
    -not (Test-Path (Join-Path $Source "installer\posetally_setup.py"))) {
    Stop-Install "The PoseTally program files were not found. Unpack the whole ZIP file first, then run Install-Windows.bat from the unpacked folder."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    Stop-Install "PoseTally needs 64-bit Windows."
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

if ($env:POSETALLY_HOME) { $AppHome = $env:POSETALLY_HOME } else { $AppHome = Join-Path $env:LOCALAPPDATA "PoseTally" }
New-Item -ItemType Directory -Force -Path (Join-Path $AppHome "bin") | Out-Null

$env:UV_PYTHON_INSTALL_DIR = Join-Path $AppHome "python"
$env:UV_CACHE_DIR = Join-Path $AppHome "cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_NO_CONFIG = "1"
$Uv = Join-Path $AppHome "bin\uv.exe"

$HaveUv = $false
if (Test-Path $Uv) {
    try {
        $Found = (& $Uv --version) 2>$null
        if ("$Found" -match [regex]::Escape(" $UvVersion")) { $HaveUv = $true }
    } catch { }
}
if (-not $HaveUv) {
    Write-Host ""
    Write-Host "Downloading the package tool (uv $UvVersion) ..."
    $Temp = Join-Path ([IO.Path]::GetTempPath()) ("posetally_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $Temp | Out-Null
    $Zip = Join-Path $Temp "uv.zip"
    $Url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
        Expand-Archive -Path $Zip -DestinationPath $Temp -Force
        $Exe = Get-ChildItem -Path $Temp -Recurse -Filter "uv.exe" | Select-Object -First 1
        if (-not $Exe) { throw "uv.exe was not in the download" }
        Copy-Item -Path $Exe.FullName -Destination $Uv -Force
    } catch {
        Remove-Item -Path $Temp -Recurse -Force -ErrorAction SilentlyContinue
        Stop-Install ("The download failed. Check the internet connection and run the installer again. (" + $_.Exception.Message + ")")
    }
    Remove-Item -Path $Temp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Installing a private copy of Python $PythonVersion (it does not touch any Python you already have) ..."
& $Uv python install $PythonVersion
if ($LASTEXITCODE -ne 0) { Stop-Install "Python could not be installed." }

$EnvDir = Join-Path $AppHome "env"
& $Uv venv --python $PythonVersion --seed --allow-existing $EnvDir
if ($LASTEXITCODE -ne 0) { Stop-Install "The Python environment could not be created." }

$Python = Join-Path $EnvDir "Scripts\python.exe"
& $Python (Join-Path $Source "installer\posetally_setup.py") --source $Source --home $AppHome --uv $Uv @args
exit $LASTEXITCODE
