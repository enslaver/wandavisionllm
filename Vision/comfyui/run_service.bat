@echo off
cd /d "%~dp0"
set "path=%windir%\System32;%windir%\System32\WindowsPowerShell\v1.0;%PATH%"
:loop
if exist ".\python_embeded\pythonw.exe" (set "PY=.\python_embeded\pythonw.exe") else (set "PY=.\python_embeded\python.exe")
start /B /WAIT %PY% -I -W ignore::FutureWarning ComfyUI\main.py --windows-standalone-build --listen 127.0.0.1 --port 8188 >> "%~dp0comfy_service.log" 2>&1
timeout /t 10 >nul
goto loop
