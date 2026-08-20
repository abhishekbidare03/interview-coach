<#
    Interview Coach - one-click launcher.

    Everything the app needs, checked and started in order, then the browser
    opened once the models are actually warm. Closing this window stops the
    server.

    The design point is that a failure should say what to do about it. Startup
    can fail four ways that all look identical from the desktop - no venv, no
    Ollama, no model pulled, no voice file - so each is checked separately and
    reported by name rather than surfacing as a Python traceback.

    Run by the desktop shortcut. Also fine to run directly:
        powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#>

$ErrorActionPreference = 'Stop'
$Root    = Split-Path -Parent $PSScriptRoot
$Python  = Join-Path $Root '.venv\Scripts\python.exe'
$Voice   = Join-Path $Root 'data\voices\en_US-lessac-high.onnx'
$AppUrl  = 'http://127.0.0.1:8000'
$Ollama  = 'http://127.0.0.1:11434'
$Model   = 'qwen2.5:3b'

$Host.UI.RawUI.WindowTitle = 'Interview Coach  -  close this window to stop'

function Say([string]$Text, [string]$Colour = 'Gray') {
    Write-Host $Text -ForegroundColor $Colour
}

function Fail([string]$What, [string]$Fix) {
    Say ''
    Say "  $What" 'Red'
    Say ''
    Say "  To fix it:" 'Yellow'
    Say "    $Fix" 'Yellow'
    Say ''
    Read-Host '  Press Enter to close'
    exit 1
}

# Small helper: is something answering on this URL? Used for both services, so
# a slow-but-alive Ollama is not mistaken for a missing one.
function Probe([string]$Url, [int]$TimeoutSec = 2) {
    try {
        $null = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing
        return $true
    } catch { return $false }
}

Say ''
Say '  Interview Coach' 'Cyan'
Say '  ---------------' 'DarkCyan'
Say ''

# -- 0. Already running? ---------------------------------------------------- #
# Starting a second server just produces a confusing "address in use" error, so
# treat an existing one as success and go straight to the browser.
if (Probe "$AppUrl/api/health") {
    Say '  Already running. Opening the browser.' 'Green'
    Start-Process $AppUrl
    Start-Sleep -Seconds 1
    exit 0
}

# -- 1. The environment ------------------------------------------------------ #
if (-not (Test-Path $Python)) {
    Fail 'The Python environment is missing.' `
         "cd `"$Root`"  then:  uv sync"
}
if (-not (Test-Path $Voice)) {
    Fail 'The Piper voice file is missing (data\voices\en_US-lessac-high.onnx).' `
         'Download en_US-lessac-high from huggingface.co/rhasspy/piper-voices'
}

# -- 2. Ollama --------------------------------------------------------------- #
Write-Host '  Ollama              ' -NoNewline
if (Probe "$Ollama/api/tags") {
    Say 'already running' 'Green'
} else {
    $exe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $exe) {
        Say 'not found' 'Red'
        Fail 'Ollama is not installed, or not on PATH.' 'Install it from ollama.com'
    }
    Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden
    $up = $false
    foreach ($i in 1..30) {
        Start-Sleep -Milliseconds 500
        if (Probe "$Ollama/api/tags" 1) { $up = $true; break }
    }
    if (-not $up) { Say 'failed to start' 'Red'; Fail 'Ollama would not start.' 'Try running: ollama serve' }
    Say 'started' 'Green'
}

# -- 3. The model ------------------------------------------------------------ #
Write-Host '  Language model      ' -NoNewline
try {
    $tags = (Invoke-WebRequest -Uri "$Ollama/api/tags" -TimeoutSec 5 -UseBasicParsing).Content |
            ConvertFrom-Json
    $names = @($tags.models | ForEach-Object { $_.name })
} catch { $names = @() }

if ($names -contains $Model) {
    Say "$Model ready" 'Green'
} else {
    Say 'not pulled' 'Red'
    Fail "The model $Model has not been downloaded." "ollama pull $Model"
}

# -- 4. Open the browser the moment the server is warm ----------------------- #
# Model loading takes roughly fifteen seconds, so opening the browser now would
# land on a connection error. A detached process waits for the health endpoint
# and opens it then.
#
# It is a separate process rather than a background job because the server below
# runs in the foreground and blocks this console until it stops. It prints
# nothing: a job's output does not reach the parent console anyway, and the
# server's own startup line already reports what matters.
$poll = @"
foreach (`$i in 1..150) {
    Start-Sleep -Milliseconds 400
    try {
        `$null = Invoke-WebRequest -Uri '$AppUrl/api/health' -TimeoutSec 2 -UseBasicParsing
        Start-Process '$AppUrl'
        break
    } catch { }
}
"@
Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
    -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $poll

Say ''
Say '  Loading models - the browser opens by itself in about 15 seconds.' 'DarkGray'
Say '  In the log below, check it says  stt=cuda  and not  stt=cpu.' 'DarkGray'

# -- 5. The server ----------------------------------------------------------- #
# Foreground on purpose. Its log is the whole point of this window, and closing
# the window is how you stop it.
Say ''
Push-Location $Root
try {
    & $Python -m coach.server
} finally {
    Pop-Location
    Say ''
    Say '  Interview Coach stopped.' 'DarkGray'
    Start-Sleep -Seconds 2
}
