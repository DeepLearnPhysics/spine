---
name: Bug report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
A clear set of commands and resources needed to reproduce the issue:
- Code
- Input file path (if any, must be at S3DF, NERSC, FNAL, ANL or public HTML)
- `Driver` configuration file path (if using the driver class)
```
# All necessary imports at the beginning
import yaml
from spine.driver import Driver

# A succinct reproducing example trimmed down to the essential parts:
cfg = yaml.safe_load('/path/to/config.cfg')
driver = Driver(cfg)
driver.run()
```
If the necessary information to reproduce the issue is not provided, the issue will not be addressed.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots, histograms, event displays**
If applicable, add figures to help explain your problem.

**System Information**
Provide the following:
 - OS: [e.g., Ubuntu 22.04, macOS 14.0, CentOS 7]
 - Python version: [e.g., 3.9.7]
 - SPINE release version (or github commit tag if not using a tagged release)
 - Key dependencies: [e.g., PyTorch 2.0.1, NumPy 1.24.3]
 - Singularity/apptainer/docker image (if applicable)

**Additional context**
Add any other context about the problem here.

**Possible solution (optional)**
Offer suggestion(s) as to how to address the issue.
