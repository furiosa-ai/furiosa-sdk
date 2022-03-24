# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [unreleased]
### Changed
- Fix the missing git_version files and README.md in some modules #318
- litmus should use furiosa-tools compile rather than session.create() #317
- More abundant version information in CLI tools #320

## [0.6.1]
### Fixed
- Compiler report shouldn't be displayed unless running as unit tests (#306)
- Throw an exception if session is already closed (#305)
