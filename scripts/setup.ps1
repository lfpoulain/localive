<#
.SYNOPSIS
  Installe Localive sous Windows : Python, llama.cpp, les modèles, et vérifie Docker.

.DESCRIPTION
  1. .venv (Python 3.12 via uv) et dépendances ; PyTorch CUDA si une carte NVIDIA est là.
  2. python -m localive.install : llama.cpp b11960 (Vulkan ou CPU), Qwen3-4B (cerveau),
     d3-mini Q8_0 (sélection des outils), Smart Turn (fin de tour), Parakeet (oreilles)
     et Phonon 40m-fr (voix). Chaque fichier
     est vérifié par SHA-256 et repris s'il est interrompu.
  Phonon 40m est à conditions : les accepter une fois sur
  https://huggingface.co/gradium/phonon-40m-fr, puis connecter Hugging Face
  (.venv\Scripts\hf.exe auth login), ou relancer plus tard sans -SkipVoice.
  Relancer le script est sans risque : ce qui est déjà vérifié est conservé.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cpu -Llama vulkan
  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -NoToolRouter
  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cpu -Llama cpu -SkipVoice
  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models gemma-e4b,qwen3.5-4b
  (-Models : autres modèles de raisonnement, proposés ensuite dans la page Cerveau ;
   noms dans docs/installation.md : qwen3.5-9b, gemma-e4b, ministral-8b, gpt-oss-20b…)
#>
param(
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [switch]$SkipVoice,
    [switch]$ToolRouter,
    [string[]]$Models = @(),
    [ValidateSet('auto', 'cpu', 'vulkan')]
    [string]$Llama = 'auto',
    [switch]$NoToolRouter
)

$ErrorActionPreference = 'Stop'
if ($ToolRouter -and $NoToolRouter) { throw 'Choisissez soit -ToolRouter soit -NoToolRouter.' }
Set-Location ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')))
# Keep the installer cache inside the project, including in restricted environments.
$env:UV_CACHE_DIR = Join-Path $PWD '.runtime\uv-cache'

Write-Host '1/2 Environnement Python'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv est requis : winget install --id=astral-sh.uv -e (puis rouvrir PowerShell). https://docs.astral.sh/uv/'
}
$python = '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE) { throw 'Création du venv Python échouée.' }
}
if ($Device -eq 'auto') {
    $Device = 'cpu'
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $Device = 'cuda' }
    }
}
if ($Device -eq 'cpu') {
    & $python -c "from importlib.util import find_spec; import sys; sys.exit(find_spec('torch') is None)"
    if ($LASTEXITCODE) {
        uv pip install --python $python --index https://download.pytorch.org/whl/cpu 'torch>=2.8'
        if ($LASTEXITCODE) { throw 'Installation de PyTorch CPU échouée.' }
    }
}
uv pip install --python $python -e .
if ($LASTEXITCODE) { throw 'Installation des dépendances Python échouée.' }
if ($Device -eq 'cuda') {
    & $python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE) {
        uv pip install --python $python torch --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
        if ($LASTEXITCODE) { throw 'Installation de PyTorch CUDA échouée.' }
    }
    & $python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'PyTorch ne voit pas la carte NVIDIA')"
    if ($LASTEXITCODE) { throw 'Une carte NVIDIA compatible CUDA est requise. Sinon : -Device cpu.' }
}

Write-Host '2/2 Moteurs et modèles'
$installArgs = @('-m', 'localive.install', '--device', $Device, '--llama', $Llama)
if ($SkipVoice) { $installArgs += '--skip-voice' }
if ($NoToolRouter -or ($PSBoundParameters.ContainsKey('ToolRouter') -and -not $ToolRouter)) {
    $installArgs += '--no-tool-router'
}
elseif ($ToolRouter) { $installArgs += '--tool-router' }
if ($Models) { $installArgs += @('--models', ($Models -join ',')) }
& $python @installArgs
if ($LASTEXITCODE) { throw 'Installation interrompue : voir le message ci-dessus, puis relancer ce script.' }

Write-Host "`nPrêt. Démarrer : double-cliquer « Lancer Localive.cmd », ou"
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts/start.ps1'
Write-Host 'Administration : http://127.0.0.1:8787'
