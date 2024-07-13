@echo off &setlocal
IF NOT EXIST tests_cpp.generated goto Create
IF NOT EXIST tests_unit.generated goto Create

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

if %var1% LSS %var2% goto Message

:Create
echo Generating Files
type nul > tests_cpp.generated 
type nul > tests_unit.generated
start C:\GameDev\ExpertSystems.AI\x64\Release\es_test_app.exe "-b" "C:\GameDev\ExpertSystems.AI\src\es_test\tests.list" "C:\GameDev\ExpertSystems.AI\src\es_test\tests_cpp.generated" "C:\GameDev\ExpertSystems.AI\src\es_test\tests_unit.generated"

:Message
echo The list has not been modified.

:End