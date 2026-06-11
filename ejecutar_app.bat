@echo off
cd /d "%~dp0"

echo ============================================
echo App final Streger - FV, BESS y finanzas
echo ============================================

if not exist .venv (
    echo Creando entorno virtual...
    py -m venv .venv
)

call .venv\Scripts\activate

echo Instalando dependencias...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

echo Iniciando aplicacion...
streamlit run app.py

pause
