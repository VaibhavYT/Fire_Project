@echo off
echo Starting Python HTTP Server for C:\Recordings
cd C:\Recordings
python -m http.server 8000
pause