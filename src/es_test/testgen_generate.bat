IF NOT EXIST tests_cpp.generated type nul > tests_cpp.generated
IF NOT EXIST tests_unit.generated type nul > tests_unit.generated

@echo off &setlocal

Set orig=tests.list
Set gen=tests_cpp.generated

Set orig_ModTime=
Set gen_ModTime=

set /a Hour = Hour %% 12
if %Hour%==0 set "Hour=12"

for /f %%i in ('"forfiles /m %orig% /c "cmd /c echo @ftime" "') do set orig_ModTime=%%i
for /f %%j in ('"forfiles /m %gen% /c "cmd /c echo @ftime" "') do set gen_ModTime=%%j

SET "var1=%orig_ModTime::=%"
SET "var2=%gen_ModTime::=%"

if %var1% LSS %var2% goto End

start D:\Gamedev\ExpertSystems.AI\x64\Release\es_test_app.exe "-b" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests.list" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests_cpp.generated" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests_unit.generated"

:End