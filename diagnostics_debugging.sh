#!/bin/bash
# This is a simple debugging script that runs with most recent SNWC data and outputs diagnostics plots to directory figures/

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
./retrieval_script.sh
cd ../..

### DEBUGGING CALLS
python2 call_interpolation.py --model_data testdata/latest/fcst_tprate.grib2 --background_data testdata/latest/mnwc_tprate.grib2 --dynamic_nwc_data testdata/latest/mnwc_tprate_full.grib2 --extrapolated_data testdata/latest/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/latest/output/smoothed_mnwc_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 9 --plot_diagnostics "yes"
python2 call_interpolation.py --model_data testdata/latest/fcst_cc.grib2 --dynamic_nwc_data testdata/latest/mnwc_cc.grib2 --output_data testdata/latest/output/smoothed_mnwc_TCC.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 9 --plot_diagnostics "yes"
python2 call_interpolation.py --obs_data testdata/latest/obs_2t.grib2 --model_data testdata/latest/fcst_2t.grib2 --output_data testdata/latest/output/interpolated_2t.grib2 --parameter Temperature --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python2 call_interpolation.py --obs_data testdata/latest/obs_2r.grib2 --model_data testdata/latest/fcst_2r.grib2 --output_data testdata/latest/output/interpolated_2r.grib2 --parameter Dewpoint --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python2 call_interpolation.py --obs_data testdata/latest/obs_msl.grib2 --model_data testdata/latest/fcst_msl.grib2 --output_data testdata/latest/output/interpolated_msl.grib2 --parameter msl --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics "yes"
python2 call_interpolation.py --obs_data testdata/latest/obs_10si.grib2 --model_data testdata/latest/fcst_10si.grib2 --output_data testdata/latest/output/interpolated_10si.grib2 --parameter wind10m --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics "yes"

cd testdata/latest/
find . -name '*.grib2' -delete
