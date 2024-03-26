# lego-alma
## Overview
The `lego-alma` program reads data from the hardware of the LEGO ALMA 2.0 interferometer model, and displays interferometric observations
on a screen. The code used in the previous iteration of the project relied heavily on the [friendly Radio Virtual Interferometer](https://crpurcell.github.io/friendlyVRI/)
developed by the Nordic ARC Node. This version, although inspired bny the fVRI project, was completely rewritten to better suit the needs
of LEGO ALMA, which requires a faster update but less initial customization.

## Installation
Copy the code on your drive. The required packages are listed in the requirements.txt file.

## Setup
The dictionary controlling the button setup can be modified in the `la_config.py` file.
