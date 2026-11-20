@echo off
rem see README.md
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\..\.."
call .venv\Scripts\activate.bat
streamlit run app.py
