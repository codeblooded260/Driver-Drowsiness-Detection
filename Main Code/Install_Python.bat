@echo off

setlocal enableDelayedExpansion

set pythonPath=%PATH:Python=%
if "%pythonPath%"=="%PATH%" (
  echo Python is not installed on this computer.
  echo Downloading latest version of Python...
  powershell -Command "& { (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe', 'python-3.10.0-amd64.exe') }"
  echo Installing Python...
  start /wait python-3.10.0-amd64.exe /quiet
  echo Python has been installed.
  del python-3.10.0-amd64.exe
) else (
  echo Python is already installed on this computer.
)

echo Done.
