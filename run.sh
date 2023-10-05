set -xue

object_exists(){
  url=$1
  code=$(curl -o /dev/null --silent -Iw '%{http_code}' $url)

  if [ $code = "200" ]; then
    return 0
  fi

  return 1
}

date

grib_opts="generatingProcessIdentifier=202"

if [ "$TYPE" = "preop" ]; then
  grib_opts="generatingProcessIdentifier=203"
fi

bucket=routines-data-$ENVIRONMENT

if [ "$CLOUD" = "aws" ]; then
  bucket="fmi-$bucket"
fi

time_offset="--time_offset 1"

input="s3://$bucket/smartmet-nwc/$TYPE/${TARGET_GEOMETRY_NAME}/${DATETIME}"
output="s3://$bucket/smartmet-nwc/$TYPE/$TARGET_GEOMETRY_NAME/${DATETIME}/interpolated_${PARAM}.grib2"

# for checking if optional data exists
# requires non-authenticated read-access to data!
sourceurl="https://$S3_HOSTNAME/$bucket/smartmet-nwc/$TYPE/${TARGET_GEOMETRY_NAME}/${DATETIME}"
 
if [ "$TARGET_GEOMETRY_NAME" != "MEPS1000D" ]; then
  time_offset=""
fi

cd /nowcasting_fcst

predictability=8

if [ "$TYPE" = "preop" ]; then
  predictability=11
fi


if [ "$PARAM" = "tprate" ]; then

  model_data_opt="--model_data        $input/fcst_${PARAM}.grib2"

  if [ "$TARGET_GEOMETRY_NAME" = "MEPS1000D" ]; then
    model_data_opt=""
  fi

  ppn_data_opt=""

  if object_exists $sourceurl/ppn_${PARAM}.grib2; then
    ppn_data_opt="--obs_data $input/ppn_${PARAM}_obs.grib2 --extrapolated_data $input/ppn_${PARAM}.grib2"
  fi

  python3 ./call_interpolation.py \
    $model_data_opt \
    --background_data    $input/mnwc_${PARAM}.grib2 \
    --dynamic_nwc_data   $input/mnwc_${PARAM}_full.grib2 \
    $ppn_data_opt \
    --output_data        $output \
    --parameter          ${PARAM} \
    --mode               model_fcst_smoothed \
    --predictability     $predictability \
    --grib_write_options $grib_opts \
    $time_offset

elif [ "${PARAM}" = "cc" ]; then
    cc_data_opt=""

    if object_exists $sourceurl/cloudcast_${PARAM}.grib2; then
      cc_data_opt="--extrapolated_data $sourceurl/${TARGET_GEOMETRY_NAME}/${DATETIME}/cloudcast_${PARAM}.grib2"
    fi

    python3 ./call_interpolation.py \
      --model_data         $input/fcst_${PARAM}.grib2 \
      --dynamic_nwc_data   $input/mnwc_${PARAM}.grib2 \
      $cc_data_opt \
      --output_data        $output \
      --parameter          total_cloud_cover \
      --mode               model_fcst_smoothed \
      --predictability     $predictability \
      --grib_write_options $grib_opts

elif [ "${PARAM}" = "tstm" ] && [ "${TYPE}" = "preop" ]; then
  tstm_data="$input/mnwc_${PARAM}.grib2"

  if object_exists $sourceurl/hrnwc_${PARAM}.grib2; then
    tstm_data="$input/hrnwc_${PARAM}.grib2"
  fi

  python3 ./call_interpolation.py \
    --model_data         $input/fcst_${PARAM}.grib2 \
    --dynamic_nwc_data   $tstm_data \
    --output_data        $output \
    --parameter          pot \
    --mode               model_fcst_smoothed \
    --predictability     10 \
    --grib_write_options $grib_opts

else
  mode="model_fcst_smoothed"
  predictability=4

  if [ "${TYPE}" = "preop" ]; then
    predictability=11
  fi

  if [ "${TYPE}" = "prod" ] && [ "${PARAM}" = "tstm" ]; then
    mode="analysis_fcst_smoothed"
  fi

  python3 ./call_interpolation.py \
    --model_data         $input/fcst_${PARAM}.grib2 \
    --dynamic_nwc_data   $input/mnwc_${PARAM}.grib2 \
    --output_data        $output \
    --parameter          ${PARAM} \
    --mode               $mode \
    --predictability     $predictability \
    --grib_write_options $grib_opts

fi

date
