IF NOT EXIST tests_cpp.generated type nul > tests_cpp.generated
IF NOT EXIST tests_unit.generated type nul > tests_unit.generated

start D:\Gamedev\ExpertSystems.AI\x64\Release\es_test_app.exe "-b" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests.list" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests_cpp.generated" "D:\Gamedev\ExpertSystems.AI\src\es_test\tests_unit.generated"
