@echo off
cd /d "%~dp0"
:loop
caddy.exe run --config "%~dp0Caddyfile" >> "%~dp0caddy.log" 2>&1
timeout /t 10 >nul
goto loop
