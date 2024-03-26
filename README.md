# lego-alma
## Overview
The `lego-alma` program reads data from the hardware of the LEGO ALMA 2.0 interferometer model, and displays interferometric observations
on a screen. It is currently used with the LEGO ALMA 2.0 model, at the Argelander Institute for Astronomy. The code was developed by Toma Bădescu.

The code used in the previous iteration of the project relied heavily on the [friendly Radio Virtual Interferometer](https://crpurcell.github.io/friendlyVRI/)
developed by the Nordic ARC Node. This version, although inspired bny the fVRI project, was completely rewritten to better suit the needs
of LEGO ALMA, which requires a faster update but less initial customization.

## Installation
Copy the code on your drive. The required packages are listed in the requirements.txt file.

## Initial Setup

You will need access to the `/dev` files or wherever the serial input files are on your system. If 
they are not in `/dev` then this path has to be given in the `la_main.py` file.
When used in conjunction with the original LEGO ALMA 2.0 Hardware, no further setup should be needed.

The dictionary controlling the button setup can be modified in the `la_config.py` file.
The bits for the buttons and switches are also hardcoded into the Controller object, and will
be used if no configuration is given when this object is initialized.
The antenna positions on the tables and their corresponding bit positions are given in the `ant_pos.2.txt` file.
The button dictionary is also stored as a pickle in `but_dict.pickle`.
The images for the display are saved in the `/models` folder. You can replace those files with other similarly
named files.

## Running

go into the location of the la_main.py file and run it using 
`sudo -E la_main.py`, or whatever command you need on your system 
that gives you access to the needed files and the python environment.
Unless you have set up the Ultra Wideband Receivers and want to use them, answer no 
when prompted about using VLBI/ Ultra Wideband Receivers.

## Using the Ulra Wideband Receivers with lego-alma to simulte VLBI observations

Setup the ultrawideband receivers, following the instructions at [Decawave](https://www.qorvo.com/products/d/da007996)
and using the [Decawave RTLS Android app](https://apkcombo.com/decawave-drtls-manager-r1/com.decawave.argomanager/). 

If using the original devices, make sure to set them accoring to the designation on their casing (tag vs anchor). When prompted about the use of these 
devices on program startup, answer yes, and run a calibration by following the instructions on the screen. Once setup is done, you don't need to run 
it again, and the saved setup file can be loaded, unless the positions of the anchors has changed. 

## Closing 

To close the application, go to the terminal window where the program is running and press `ctrl-C`.

##License

[MIT License](LICENSE)




