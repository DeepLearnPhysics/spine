"""Simple module which define logging module style and returns it."""

import sys
import logging
import warnings

# Configure the formatting of the logger
logging.basicConfig(format='%(message)s', stream=sys.stdout)
#logging.basicConfig(format='[%(levelname)s] %(message)s')
#logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s')

# Capture warning messages and redirect them through the logger
logging.captureWarnings(True)

# Initialize logger
logger = logging.getLogger('mlreco')

# Configure the warnings package to only issue warnings once
warnings.simplefilter('once')
