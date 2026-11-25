<#
.SYNOPSIS
  Démarre Localive et ouvre sa page d'administration quand il est prêt.

.DESCRIPTION
  Le bouton « Redémarrer » de l'administration arrête Localive avec le code 75 :
  ce script le relance aussitôt. Toute autre fin (Ctrl+C, erreur) arrête la boucle.
  Si Localive tourne déjà, ce script ouvre simplement son administration.
  Journal : .runtime\logs\localive.log

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/start.ps1
  powershell -ExecutionPolicy Bypass -File scripts/start.ps1 -NoBrowser
#>
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
Set-Location ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')))
$python = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Installation manquante : powershell -ExecutionPolicy Bypass -File scripts/setup.ps1'
}

# Already running? (a second copy would only fail on the busy port)
$port = 8787
if (Test-Path localive.toml) {
    $inServer = $false
    foreach ($line in Get-Content localive.toml) {
        if ($line -match '^\s*\[(.+)\]') { $inServer = $Matches[1] -eq 'server' }
        elseif ($inServer -and $line -match '^\s*port\s*=\s*(\d+)') { $port = [int]$Matches[1]; break }
    }
}
try {
    $health = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2
    Write-Host "Localive tourne déjà ($($health.status)) : ouverture de l'administration."
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$port" }
    exit 0
} catch { }

$launchArgs = @('-m', 'localive')
if (-not $NoBrowser) { $launchArgs += '--open' }
while ($true) {
    & $python @launchArgs
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 1 }  # Python did not even start
    if ($code -ne 75) { exit $code }
    Write-Host 'Redémarrage demandé depuis l''administration…'
    $launchArgs = @('-m', 'localive')  # the page is already open
}
