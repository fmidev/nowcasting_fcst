#!/bin/bash


################## MAIN ##################

####### General parameters
PYTHON=${PYTHON:-'/home/users/ylhaisi/python-virtualenvs/nowcasting_fcst/bin/python'}          # /fmi/dev/python_virtualenvs/venv/bin/python'} # This environment was used at dev.elmo (run using "source /fmi/dev/python_virtualenvs/venv/bin/activate")
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
