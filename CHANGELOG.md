# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [unreleased]
### Changed
- Fix the missing git_version files and README.md in some modules #318
- More abundant version information in CLI tools #320
- litmus should use furiosa-tools compile rather than session.create() #317
- Change the default device to npu0pe0-1 (fusioned 2pe)
- Replace setuptools with flit to follow PE517 compliant structure #270
- Simplify checking if a model is partially quantized #349

## [0.6.1]
### Fixed
- Compiler report shouldn't be displayed unless running as unit tests (#306)
- Throw an exception if session is already closed (#305)
