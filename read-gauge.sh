#!/bin/bash

today=$(date +'%Y-%m-%d')
img_file_name=gauge-${today}.jpg

# take picture with raspberry pi camera
raspistill -o images/${img_file_name}

# for local testing without real images
#cp images/gauge-23.jpg images/${img_file_name}

# read the value of the analog gauge image
python3 analog_gauge_reader.py --image images/${img_file_name}