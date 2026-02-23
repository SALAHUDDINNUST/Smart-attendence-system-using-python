@echo off
echo ========================================
echo DELETE ALL EMPLOYEES
echo ========================================
echo.
echo WARNING: This will delete all employees!
echo.
set /p confirm="Type YES to confirm: "
if "%confirm%"=="YES" goto delete
echo Deletion cancelled.
pause
exit

:delete
echo.
echo Deleting employees folder...
if exist employees rmdir /s /q employees
echo Deleting model files...
if exist face_model.yml del face_model.yml
if exist labels.pkl del labels.pkl
echo.
echo Creating new employees folder...
mkdir employees
echo.
echo Done!
pause