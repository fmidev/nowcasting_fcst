#!/bin/bash
# This is a simple debugging script that takes the hour of the day as an input and runs the program through using the corresponding testdata
# The scripts assumes that the testdata is located at nowcasting_fcst/testdata/$HOD


# PARAMETER INPUTS: $1 is the HOD (hour of the day), $2 is yes/no (plot_diagnostics)

HOD=$1
PLOT_DIAGNOSTICS=$2

echo $HOD
echo $PLOT_DIAGNOSTICS

### MAKE DIRECTORIES
#mkdir -p figures/fields
#mkdir -p figures/jumpiness_absdiff
#mkdir -p figures/jumpiness_meandiff
#mkdir -p figures/jumpiness_ratio
#mkdir -p figures/linear_change
#mkdir -p figures/linear_change3h
#mkdir -p figures/linear_change4h

### RETRIEVE LATEST DATA
#cd testdata/latest/
#./retrieval_script.sh
#cd ../..

### DEBUGGING CALLS
python2 call_interpolation.py --model_data testdata/$HOD/fcst_tprate.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/$HOD/output/smoothed_mnwc_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 9 --plot_diagnostics $PLOT_DIAGNOSTICS
python2 call_interpolation.py --model_data testdata/$HOD/fcst_cc.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_cc.grib2 --output_data testdata/$HOD/output/smoothed_mnwc_TCC.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 9 --plot_diagnostics $PLOT_DIAGNOSTICS
python2 call_interpolation.py --obs_data testdata/$HOD/obs_2t.grib2 --model_data testdata/$HOD/fcst_2t.grib2 --output_data testdata/$HOD/output/interpolated_2t.grib2 --parameter Temperature --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
python2 call_interpolation.py --obs_data testdata/$HOD/obs_2r.grib2 --model_data testdata/$HOD/fcst_2r.grib2 --output_data testdata/$HOD/output/interpolated_2r.grib2 --parameter Dewpoint --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
python2 call_interpolation.py --obs_data testdata/$HOD/obs_msl.grib2 --model_data testdata/$HOD/fcst_msl.grib2 --output_data testdata/$HOD/output/interpolated_msl.grib2 --parameter msl --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
python2 call_interpolation.py --obs_data testdata/$HOD/obs_10si.grib2 --model_data testdata/$HOD/fcst_10si.grib2 --output_data testdata/$HOD/output/interpolated_10si.grib2 --parameter wind10m --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS

#cd testdata/latest/
#find . -name '*.grib2' -delete
