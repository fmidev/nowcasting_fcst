#!/bin/bash
# This is a simple debugging script that runs with most recent SNWC data and outputs diagnostics plots to directory figures/
alias python='python3'

### MAKE DIRECTORIES
mkdir -p figures/fields
mkdir -p figures/jumpiness_absdiff
mkdir -p figures/jumpiness_meandiff
mkdir -p figures/jumpiness_ratio
mkdir -p figures/linear_change
mkdir -p figures/linear_change3h
mkdir -p figures/linear_change4h

### RETRIEVE LATEST DATA
cd testdata/latest/
find . -name '*.grib2' -delete
./retrieval_script.sh
cd ../..

### DEBUGGING CALLS
python3 ./call_interpolation.py --model_data testdata/latest/fcst_10si.grib2 --dynamic_nwc_data testdata/latest/mnwc_10si.grib2 --output_data testdata/latest/output/interpolated_10si.grib2 --parameter 10si --mode model_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python3 ./call_interpolation.py --model_data testdata/latest/fcst_2r.grib2 --dynamic_nwc_data testdata/latest/mnwc_2r.grib2 --output_data testdata/latest/output/interpolated_2r.grib2 --parameter 2r --mode model_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python3 ./call_interpolation.py --model_data testdata/latest/fcst_2t.grib2 --dynamic_nwc_data testdata/latest/mnwc_2t.grib2 --output_data testdata/latest/output/interpolated_2t.grib2 --parameter 2t --mode model_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python3 ./call_interpolation.py --model_data testdata/latest/fcst_cc.grib2 --dynamic_nwc_data testdata/latest/mnwc_cc.grib2 --output_data testdata/latest/output/interpolated_cc.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 8 --plot_diagnostics "yes"
python3 ./call_interpolation.py --obs_data testdata/latest/ppn_tprate_obs.grib2 --model_data testdata/latest/fcst_tprate.grib2 --background_data testdata/latest/mnwc_tprate.grib2 --dynamic_nwc_data testdata/latest/mnwc_tprate_full.grib2 --extrapolated_data testdata/latest/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/latest/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics "yes"

cd testdata/latest/
find . -name '*.grib2' -delete
