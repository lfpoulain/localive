@echo off
rem Double-clic : démarre Localive et ouvre son administration.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
rem A native crash returns a negative code: "if errorlevel 1" would miss it.
if not "%errorlevel%"=="0" (
  echo.
  echo Localive s'est arrete sur une erreur ^(code %errorlevel%^).
  echo Le detail est ci-dessus et dans .runtime\logs\localive.log
  pause
)
