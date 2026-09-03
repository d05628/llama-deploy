@echo off
cd /d "%~dp0"

:: 1. 异步启动 Python 服务
start "" python manager.py

:: 2. 等待 3 秒确保服务起来了
timeout /t 3 /nobreak >nul

:: 3. 指定用 Edge 浏览器打开网页
start http://localhost:9090

exit