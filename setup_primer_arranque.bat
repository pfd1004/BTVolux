@echo off
cd /d "%~dp0"

echo Configurando entornos portables...
echo.

if exist "runtime\env_app\Scripts\conda-unpack.exe" (
    echo Configurando env_app...
    runtime\env_app\Scripts\conda-unpack.exe
)

if exist "runtime\env_nnunet\Scripts\conda-unpack.exe" (
    echo Configurando env_nnunet...
    runtime\env_nnunet\Scripts\conda-unpack.exe
)

if exist "runtime\env_agunet\Scripts\conda-unpack.exe" (
    echo Configurando env_agunet...
    runtime\env_agunet\Scripts\conda-unpack.exe
)

if exist "runtime\env_plsnet\Scripts\conda-unpack.exe" (
    echo Configurando env_plsnet...
    runtime\env_plsnet\Scripts\conda-unpack.exe
)

echo.
echo Configuracion terminada.
pause