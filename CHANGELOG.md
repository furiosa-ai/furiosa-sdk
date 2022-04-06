# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Unreleased]
### Added

### Changed

## [0.7.0]
### Added
- Session should support batch size and compiler_hints flag #375
- Profiler API added to enable/disable runtime profiler dynamically #363
- Adopt the profiler enabled by env variable to async session #393

### Changed
- Fix the missing git_version files and README.md in some modules #318
- More abundant version information in CLI tools #320
- litmus should use furiosa-tools compile rather than session.create() #317
- Change the default device to npu0pe0-1 (fusioned 2pe)
- Replace setuptools with flit to follow PE517 compliant structure #270
- Simplify checking if a model is partially quantized #349
- NUX_PROFILER_PATH environment deprecated via new FURIOSA_PROFILER_OUTPUT_PATH #363
- Furiosa Model exposes blocking API over non-blocking API #414
- Upgrade ONNX OperatorSet version to 13 #131

## [0.6.1]
### Fixed
- Compiler report shouldn't be displayed unless running as unit tests (#306)
- Throw an exception if session is already closed (#305)
