@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

REM Check for specific targets that need special handling
if "%1" == "apidoc" goto apidoc
if "%1" == "html" goto html

REM Default: pass through to Sphinx
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:apidoc
echo.Generating API documentation...
REM -f: force overwrite
REM -e: put each module file in its own page (no single big file)
REM -M: put module documentation before submodule documentation
REM -o: output directory (source/reference)
sphinx-apidoc -f -e -M -o source/reference ../src/pytrebuchet
if "%1" == "apidoc" goto end
exit /b

:html
call :apidoc
echo.Building HTML documentation...
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
