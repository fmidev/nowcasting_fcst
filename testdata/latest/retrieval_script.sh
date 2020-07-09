#!/bin/bash

wget http://scout.fmi.fi/tmp/smartmet-nwc.tar.gz --proxy=off
tar xvfz smartmet-nwc.tar.gz
mkdir -p output/
rm smartmet-nwc.tar.gz

