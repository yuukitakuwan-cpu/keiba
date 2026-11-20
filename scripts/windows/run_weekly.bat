@echo off
rem see README.md
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\..\.."
if not exist data\logs mkdir data\logs
call .venv\Scripts\activate.bat
python -m keiba weekly >> data\logs\weekly.log 2>&1
