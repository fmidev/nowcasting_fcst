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

#search_dir=/home/korpinen/Documents/STU_kehitys/ukkosen_tod/data/2022
#STRING="0h"
#IFS='/'
#for entry in "$search_dir"/*;do
#  if [[ "$entry" != *"$STRING"* ]];then
#    model="${entry::-6}"
#    read -ra ADDR <<< "$model"
#      date="${ADDR[-1]}"
#    python3 ./call_interpolation.py --model_data "$model".grib2 --obs_data "$model"_0h.grib2 --output_data testdata/$HOD/output/interpolated_pot_"$date".grib2 --parameter total_cloud_cover --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#  fi
#done


### RETRIEVE LATEST DATA
#cd testdata/latest/
#./retrieval_script.sh
#cd ../..

### DEBUGGING CALLS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_10si.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_10si.grib2 --output_data testdata/$HOD/output/interpolated_10si.grib2 --parameter 10si --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python ./call_interpolation.py --model_data testdata/$HOD/fcst_2r.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_2r.grib2 --output_data testdata/$HOD/output/interpolated_2r.grib2 --parameter 2r --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
python ./call_interpolation.py --model_data testdata/$HOD/fcst_2t.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_2t.grib2 --output_data testdata/$HOD/output/interpolated_2t.grib2 --parameter 22t --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_cc.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_cc.grib2 --output_data testdata/$HOD/output/interpolated_cc.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --model_data testdata/$HOD/fcst_tprate.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/kaikki_t.grib2 --obs_data testdata/$HOD/Pot_oh.grib2 --output_data testdata/$HOD/output/interpolated_pot.grib2 --parameter total_cloud_cover --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python ./call_interpolation.py --dynamic_nwc_data testdata/$HOD/mnwc_cc.grib2 --extrapolated_data testdata/$HOD/mnwc_lcc3.grib2 --output_data testdata/$HOD/testinterpolated_cc.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 3 --plot_diagnostics $PLOT_DIAGNOSTICS


# Old code part, came with package
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

#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_10si.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_10si.grib2 --output_data testdata/$HOD/output/interpolated_10si.grib2 --parameter 10si --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_2r.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_2r.grib2 --output_data testdata/$HOD/output/interpolated_2r.grib2 --parameter 2r --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_2t.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_2t.grib2 --output_data testdata/$HOD/output/interpolated_2t.grib2 --parameter 2t --mode model_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_cc.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_cc.grib2 --output_data testdata/$HOD/output/interpolated_cc.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --model_data testdata/$HOD/fcst_tprate.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --detectability_data testdata/radar_detectability_field_255_280.h5 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --model_data testdata/$HOD/fcst_tprate.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS --time_offset 1

#python3 ./call_interpolation.py --obs_data testdata/$HOD/ppn_tprate_obs.grib2 --background_data testdata/$HOD/mnwc_tprate.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_tprate_full.grib2 --extrapolated_data testdata/$HOD/ppn_tprate.grib2 --output_data testdata/$HOD/output/interpolated_tprate.grib2 --parameter precipitation_1h_bg --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS --time_offset 1
#python3 ./call_interpolation.py --obs_data testdata/$HOD/POT_addmnwc1.grib2 --model_data testdata/$HOD/POTmnwc1.grib2 --output_data testdata/$HOD/output/interpolated_pot.grib2 --parameter pot --mode analysis_fcst_smoothed --predictability 4 --plot_diagnostics $PLOT_DIAGNOSTICS
#python3 ./call_interpolation.py --model_data testdata/$HOD/fcst_cc.grib2 --dynamic_nwc_data testdata/$HOD/mnwc_cc.grib2 --extrapolated_data testdata/$HOD/mnwc_lcc3.grib2 --output_data testdata/$HOD/output/interpolated_cc.grib2 --parameter total_cloud_cover --mode model_fcst_smoothed --predictability 8 --plot_diagnostics $PLOT_DIAGNOSTICS

#cd testdata/latest/
#find . -name '*.grib2' -delete
