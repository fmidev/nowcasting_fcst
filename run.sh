#!/bin/bash
# This script interpolates data between two fields (analysis and forecast) separated by $PREDICTABILITY.
# Here, predictability is taken as a constant value but in practice should be given as a forecast/flow-dependent parameter to this program, calculated in a separate module (or inside this program). In other words, analysis (obsdata) corresponds to the latest LAPS analysis whereas forecast field (modeldata) is the full-length edited forecast. The first timestamp of the edited forecast corresponds to the timestamp of the analysis field. The beginning part of the edited forecast up until $PREDICTABILITY-1 hours is replaced with the analysis-blended forecast. If forecast steps in the modeldata and $SECONDS_BETWEEN_STEPS are defined in even hours, the interpolated data should have the same time length as modeldata.

# The testdata files were retrieved from Smartmet Server using commands

# wget -O testdata/obsdata_3.nc --no-proxy 'http://smartmet.fmi.fi/download?param=Pressure&producer=laps_skandinavia&format=netcdf&starttime=201810180200&endtime=201810180200&timestep=60&projection=stereographic&centrallongitude=20,centrallatitude=90,truelatitude=60&bbox=5,55,40,71&gridsize=222,236'
# nccopy -k 4 testdata/obsdata_3.nc testdata/obsdata.nc
# rm testdata/obsdata_3.nc
# AND
# wget -O testdata/modeldata_3.nc --no-proxy 'http://smartmet.fmi.fi/download?param=Pressure&producer=pal_skandinavia&format=netcdf&starttime=201810180200&timestep=60&projection=stereographic&centrallongitude=20,centrallatitude=90,truelatitude=60&bbox=5,55,40,71&gridsize=222,236'
# nccopy -k 4 testdata/modeldata_3.nc testdata/modeldata.nc
# rm testdata/modeldata_3.nc

# => laps_skandinavia -grid needs to have a similar spatial resolution+projection than edited data (pal_skandinavia) => the max available edited resolution. The grid definition needs to be retrieved from the modeldata.nc file
# In this example, a $PREDICTABILITY of 4 hours is used.

# Known issues in code:
# in function read_nc no kind of error checking of the data is being done atm. The min/max values are taken from the raw data fields as provided. A named list for each parameter? Should here be some error checking based on plausible min/max values? missingvalue -checking already exist in the function read_nc.
# nodata fields can (in principle) be different between the timesteps.
# predictability is a constant value atm whereas in reality it has both flow- and variable-dependence.
# Value gaussian_filter_sigma is also an ad-hoc constant value atm, whereas it should depend on the spatial variability difference of the two fields
# PARAMETERS used are Pressure Temperature DewPoint Humidity WindSpeedMS (currently available in LAPS-analysis)


################## MAIN ##################

####### General parameters
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'} # This environment was used at dev.elmo (run using "source /fmi/dev/python_virtualenvs/venv/bin/activate")
DATAPATH=${DATAPATH:-"testdata/"}
OBSDATA=${OBSDATA:-"obsdata.nc"}
MODELDATA=${MODELDATA:-"modeldata.nc"}
OUTPATH=${OUTPATH:-"outdata/"}
OUTFILE_INTERP=${OUTFILE_INTERP:-"nowcast_interpolated.nc"}
PREDICTABILITY=${PREDICTABILITY:-"4"} # This is an even number in hours.
SECONDS_BETWEEN_STEPS=${SECONDS_BETWEEN_STEPS:-3600} # Edited data has a maximum temporal resolution of one hour. For illustrative purposes of the algorithm, even higher resolution can be used.
MINUTES_BETWEEN_STEPS=$(expr $SECONDS_BETWEEN_STEPS / 60)
PARAMETER=${PARAMETER:-"PRESSURE"}
MODE=${MODE:-'fcst'} # forecast mode: only one timestep from each datas is allowed as an input to call_interpolation.py

# Make output directory if not existing
mkdir -p $OUTPATH

# Updated call with additional parameters
cmd=$PYTHON" call_interpolation.py --obsdata "$DATAPATH$OBSDATA" --modeldata "$DATAPATH$MODELDATA" --seconds_between_steps "$SECONDS_BETWEEN_STEPS" --interpolated_data "$OUTPATH$OUTFILE_INTERP" --predictability "$PREDICTABILITY" --parameter "$PARAMETER" --mode "$MODE
eval $cmd
 


exit
