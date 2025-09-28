# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2025-09-27

### Fixed
- **Stopwatch timing inconsistencies**: Added validation in Driver.process() to prevent timing conflicts
- **"Cannot stop a watch more than once" errors**: Fixed Time object comparisons using value checks instead of identity
- **Iteration errors in StopwatchManager**: Fixed parameter handling in reset() method
- **Dimension scaling in visualization**: Corrected pixel coordinate conversion in vis/geo.py

### Changed
- **Driver import optimization**: Moved torch utilities to top-level imports for cleaner code structure
- **Eliminated conditional imports**: Reduced code complexity by removing scattered conditional imports

### Added
- **Stopwatch state properties**: Added `running` and `paused` properties for clean state checking
- **Enhanced documentation**: Professional styling, comprehensive API docs, and SPINE logo integration
- **Reset functionality**: Proper stopwatch reinitialization using clean instance replacement

## [0.7.0] - Previous Release

Initial release with core SPINE functionality.

---

**Note**: For detailed commit history, see the [GitHub repository](https://github.com/DeepLearnPhysics/spine).
