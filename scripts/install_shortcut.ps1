<#
    Create the "Interview Coach" desktop shortcut.

    Run once:
        powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1

    Adds -Uninstall to remove it again.

    The shortcut launches PowerShell with -ExecutionPolicy Bypass rather than
    running start.ps1 directly. Double-clicking a .ps1 opens it in Notepad on a
    default Windows install, and an unsigned script will not run at all under
    the default policy - the bypass applies to this one process only, not to the
    machine.
#>

param([switch]$Uninstall)

$ErrorActionPreference = 'Stop'

$Root     = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $Root 'scripts\start.ps1'
$Icon     = Join-Path $Root 'assets\coach.ico'
$Name     = 'Interview Coach.lnk'

# OneDrive-redirected Desktops are the norm now, so ask Windows where it is
# rather than assuming the folder under the user profile.
$Desktop  = [Environment]::GetFolderPath('Desktop')
$Link     = Join-Path $Desktop $Name
$StartDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$StartLnk = Join-Path $StartDir $Name

if ($Uninstall) {
    foreach ($p in @($Link, $StartLnk)) {
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $p" }
    }
    Write-Host 'Done.' -ForegroundColor Green
    exit 0
}

foreach ($required in @($Launcher, $Icon)) {
    if (-not (Test-Path $required)) {
        Write-Host "Missing: $required" -ForegroundColor Red
        if ($required -eq $Icon) {
            Write-Host 'Generate it first:  python scripts\make_icon.py' -ForegroundColor Yellow
        }
        exit 1
    }
}

$shell = New-Object -ComObject WScript.Shell

foreach ($target in @($Link, $StartLnk)) {
    $sc = $shell.CreateShortcut($target)
    $sc.TargetPath       = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $sc.Arguments        = "-NoProfile -ExecutionPolicy Bypass -File `"$Launcher`""
    $sc.WorkingDirectory = $Root
    $sc.IconLocation     = "$Icon,0"
    $sc.Description      = 'Local AI interview coach - starts everything and opens the browser'
    $sc.WindowStyle      = 1          # normal; the console is the stop button
    $sc.Save()
    Write-Host "created $target" -ForegroundColor Green
}

Write-Host ''
Write-Host '  Double-click "Interview Coach" on your desktop.' -ForegroundColor Cyan
Write-Host '  It starts Ollama and the server, then opens the browser when ready.'
Write-Host '  Close the console window to stop it.'
Write-Host ''
