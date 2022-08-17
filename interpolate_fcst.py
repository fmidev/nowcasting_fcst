"""Methods for spatial and temporal interpolation of precipitation fields."""
# This is slightly modified from interpolation.py: input fields can have several timesteps to them.
# This script does the interpolation between images.




try:
  import cv2
except ImportError:
  raise Exception("OpenCV Python bindings not found")
from numpy import arange, dstack, float32, isfinite, log, logical_and, meshgrid, \
  nan, ones, ubyte
import numpy as np
from scipy.ndimage import gaussian_filter

def linear(obsfields, modelfields, mask_nodata, predictability, seconds_between_steps, missingval):
  """Temporal interpolation between two precipitation fields by using advection 
  field. The motion is estimated by using the Farneback algorithm implemented 
  in OpenCV.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  NOT USED n_interp_frames : int
    Number of frames to interpolate between the given precipitation fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  farneback_params : tuple
    Parameters for the Farneback optical flow algorithm, see the documentation 
    of the Python OpenCV interface.
  R_min : float
    Minimum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_min is the min value of all values contained in R1 and R2.
  R_max : float
    Maximum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_max is the max value of all values contained in R1 and R2.
  missingval : float
    Value that is used for missing data. No interpolation is done for missing 
    values.
  logtrans : bool
    If True, logarithm is taken from R1 and R2 when computing the motion 
    vectors. This might improve the reliability of motion estimation.
  predictability : int
    Predictability in hours
  seconds_between_steps: int
    How long should two timesteps differ to each other?
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time.
  """
  
  R1 = obsfields[0,:,:]
  R2 = modelfields[predictability,:,:]
  n_interp_frames = predictability*3600/seconds_between_steps - 1 # predictability is defined in hours
  tws = 1.0*arange(1, n_interp_frames + 1) / (n_interp_frames + 1)
  
  # Initializing the result list with the modelfields.shape and obsfields[0] data
  shapes = list(modelfields.shape)
  shapes[0] = predictability+1
  R_interp = ones(shapes)
  R_interp[0,:,:] = R1

  if R1.shape != R2.shape:
    raise ValueError("R1 and R2 have different shapes")

  if (np.sum(R1!=missingval)==0) | (np.sum(R2!=missingval)==0):
    print("either data field used for interpolation contains only missing values! Cannot interpolate and replacing intermediate timesteps with modelfield values!")
    R_interp[1:,:,:] = modelfields[1:,:,:]
    # Masking in the previous step does not take into account that missing values can actually spread and dilute to actual field. Therefore appylying a simple check to remove spurious values from fields, based on absolute maximum/minimum values in either field.
    R_interp[R_interp>(R_max*1.02)] = missingval
    R_interp[R_interp<(R_min*0.98)] = missingval

    return R_interp
  
  for tw in tws:
    
    # MASK1  = logical_and(isfinite(R1), R1 != missingval)
    # MASK2  = logical_and(isfinite(R2), R2 != missingval)
    # MASK12 = logical_and(MASK1, MASK2)
    MASK12 = ~np.ma.getmask(mask_nodata)

    R_interp_cur = ones(R1.shape) * missingval
    # Linear cross-dissolve is an extremely simple linear weighting of the datasets
    R_interp_cur[MASK12] = (tw)*R2[MASK12] + (1.0-tw)*R1[MASK12]
    # MASK_ = logical_and(MASK1, ~MASK2)
    # R_interp_cur[MASK_] = R1[MASK_]
    # MASK_ = logical_and(MASK2, ~MASK1)
    # R_interp_cur[MASK_] = R2[MASK_]
    # Adding interpolated image to list
    R_interp[(int(((tws==tw).nonzero())[0])+1),:,:] = (R_interp_cur)
  
  # To analysis and interpolated images, the remaining model fields are added. If the argument "seconds_between_steps" is in even hours the result list will have the same length as input argument "modelfields"
  R_interp[predictability:,:,:] = modelfields[predictability:,:,:]
  
  return R_interp
















def advection(obsfields, modelfields, mask_nodata, farneback_params, predictability, seconds_between_steps, R_min=0.1, R_max=30.0, missingval=nan, logtrans=False):
  """Temporal interpolation between two fields by using advection 
  field. The motion is estimated by using the Farneback algorithm implemented 
  in OpenCV.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  NOT USED n_interp_frames : int
    Number of frames to interpolate between the given precipitation fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  farneback_params : tuple
    Parameters for the Farneback optical flow algorithm, see the documentation 
    of the Python OpenCV interface.
  R_min : float
    Minimum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_min is the min value of all values contained in R1 and R2.
  R_max : float
    Maximum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_max is the max value of all values contained in R1 and R2.
  missingval : float
    Value that is used for missing data. No interpolation is done for missing 
    values.
  logtrans : bool
    If True, logarithm is taken from R1 and R2 when computing the motion 
    vectors. This might improve the reliability of motion estimation.
  predictability : int
    Predictability in hours
  seconds_between_steps: int
    How long should two timesteps differ to each other?
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time, having the size "predictability".
     The first time is always R1 and the last time the model field
  """

  # OBS! modelfields contains also analysis time step as step zero!
  R1 = obsfields[1,:,:]
  R2 = modelfields[predictability,:,:]
  n_interp_frames = predictability*3600/seconds_between_steps - 1 # predictability is defined in hours
  tws = 1.0*arange(1, n_interp_frames + 1) / (n_interp_frames + 1)

  # Initializing the result list with the modelfields.shape and obsfields[0] data
  shapes = list(modelfields.shape)
  R_interp = ones(shapes)
  R_interp[0,:,:] = R1

  if (np.sum(R1!=missingval)==0) | (np.sum(R2!=missingval)==0):
    print("either data field used for interpolation contains only missing values! Cannot interpolate and replacing intermediate timesteps with modelfield values!")
    R_interp[1:,:,:] = modelfields[1:,:,:]
    # Masking in the previous step does not take into account that missing values can actually spread and dilute to actual field. Therefore appylying a simple check to remove spurious values from fields, based on absolute maximum/minimum values in either field.
    R_interp[R_interp>(R_max*1.02)] = missingval
    R_interp[R_interp<(R_min*0.98)] = missingval
    
    return R_interp

  
  if R1.shape != R2.shape:
    raise ValueError("R1 and R2 have different shapes")
  
  X,Y = meshgrid(arange(R1.shape[1]), arange(R1.shape[0]))
  W = dstack([X, Y]).astype(float32)
  
  # Here changing to float to ubyte type, for AMV calculation. ubyte is an 8-bit unsigned integral data type (http://x10.sourceforge.net/x10doc/2.3.0/x10/lang/UByte.html), the 8-bit conversion of the data is necessary for the function cv2.calcOpticalFlowFarneback. Also smearing out image as defined by https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
  R1_f = _filtered_ubyte_image(R1, mask_nodata, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)
  R2_f = _filtered_ubyte_image(R2, mask_nodata, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)
  
  if int(cv2.__version__.split('.')[0]) == 2:
    VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, *farneback_params)
    VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, *farneback_params)
  else:
    VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, None, *farneback_params)
    VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, None, *farneback_params)
    
  # For actual interpolation, original vectors V1 and V2 are used!
  for tw in tws:
    R1_warped = cv2.remap(R1, W-tw*VF,       None, cv2.INTER_LINEAR)
    R2_warped = cv2.remap(R2, W-(1.0-tw)*VB, None, cv2.INTER_LINEAR)
    
    MASK1  = logical_and(isfinite(R1_warped), R1_warped != missingval)
    MASK2  = logical_and(isfinite(R2_warped), R2_warped != missingval)
    MASK12 = logical_and(MASK1, MASK2)
    
    R_interp_cur = ones(R1_warped.shape) * missingval
    R_interp_cur[MASK12] = (1.0-tw)*R1_warped[MASK12] + tw*R2_warped[MASK12]
    MASK_ = logical_and(MASK1, ~MASK2)
    R_interp_cur[MASK_] = R1_warped[MASK_]
    MASK_ = logical_and(MASK2, ~MASK1)
    R_interp_cur[MASK_] = R2_warped[MASK_]
    # Adding interpolated image to list
    R_interp[(int(((tws==tw).nonzero())[0])+1),:,:] = (R_interp_cur)
  
  # To analysis and interpolated images, the remaining model fields are added. If the argument "seconds_between_steps" is in even hours the result list will have the same length as input argument "modelfields"
  R_interp[predictability:,:,:] = modelfields[predictability:,:,:]
  
  # Masking in the previous step does not take into account that missing values can actually spread and dilute to actual field. Therefore appylying a simple check to remove spurious values from fields, based on absolute maximum/minimum values in either field.
  R_interp[R_interp>(R_max*1.02)] = missingval
  R_interp[R_interp<(R_min*0.98)] = missingval
  
  return R_interp











def model_smoothing(obsfields, modelfields, mask_nodata, farneback_params, seconds_between_steps, predictability=None, R_min=0.1, R_max=30.0, missingval=nan, logtrans=False, sigmoid_steepness=-3.5):
  """Temporal interpolation between two fields, where motion vectors are calculated separately for each field. The weight coefficients of the fcst1 and fcst2 follows an S-curve which is defined based on data length.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  NOT USED n_interp_frames : int
    Number of frames to interpolate between the given precipitation fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  farneback_params : tuple
    Parameters for the Farneback optical flow algorithm, see the documentation 
    of the Python OpenCV interface.
  R_min : float
    Minimum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_min is the min value of all values contained in R1 and R2.
  R_max : float
    Maximum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_max is the max value of all values contained in R1 and R2.
  missingval : float
    Value that is used for missing data. No interpolation is done for missing 
    values.
  logtrans : bool
    If True, logarithm is taken from R1 and R2 when computing the motion 
    vectors. This might improve the reliability of motion estimation.
  predictability : int
    Predictability in hours. Forecast length "predictability" is completely model-based -field whereas (predictability-1) hours is a composite!
  seconds_between_steps: int
    How long should two timesteps differ to each other?
  sigmoid_steepness: float
    Defines parameter for the weighting function

  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time, having the size "predictability".
     The first time step is always obsfield and the last smoothed time step (forecast length predictability) modelfield
  """

  if predictability == None:
    raise ValueError("You need to explicitly pass a predictability value to this function!")
  # Do the motion vector calculation for each time step separately (using whole forecast length, PREDICTABILITY)
  all_fcst_lengths = list(range(1,predictability+1)) # range(1,(obsfields.shape[0])) Here, first forecast length has a non-zero coefficient and the last forecast length coefficient of one (only modelfield is used).
  n_interp_frames = len(all_fcst_lengths) # predictability is defined in hours
  # Here, a normalized half sigmoid function is used instead of linear
  #tws = 1.0*arange(1, n_interp_frames + 1) / (n_interp_frames + 1)
  #tws = sigmoid_array(np.linspace(sigmoid_steepness, -0.4, n_interp_frames))/0.5
  tws = logistic_growth_function(np.linspace(sigmoid_steepness, -0.7, n_interp_frames)) / 0.5

  # Initializing the result list with the modelfields.shape and obsfield[0] data
  shapes = list(modelfields.shape)
  R_interp = np.copy(modelfields)
  R_interp[0,:,:] = obsfields[0,:,:]
  
  
  for fcst_length in all_fcst_lengths:
      
      R1 = obsfields[fcst_length,:,:]
      R2 = modelfields[fcst_length,:,:]

      # Using only one time step
      tw=tws[all_fcst_lengths.index(fcst_length)]

      if R1.shape != R2.shape:
        raise ValueError("R1 and R2 have different shapes")

      if (np.sum(R1!=missingval)==0) | (np.sum(R2!=missingval)==0):
        print("either data field for forecast length " + str(fcst_length) + " contains only missing values!")
        R_interp[(int(((tws==tw).nonzero())[0])+1),:,:] = R2
        continue
        

      X,Y = meshgrid(arange(R1.shape[1]), arange(R1.shape[0]))
      W = dstack([X, Y]).astype(float32)

      # Here changing to float to ubyte type, for AMV calculation. ubyte is an 8-bit unsigned integral data type (http://x10.sourceforge.net/x10doc/2.3.0/x10/lang/UByte.html), the 8-bit conversion of the data is necessary for the function cv2.calcOpticalFlowFarneback. Also smearing out image as defined by https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
      R1_f = _filtered_ubyte_image(R1, mask_nodata, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)
      R2_f = _filtered_ubyte_image(R2, mask_nodata, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)

      if int(cv2.__version__.split('.')[0]) == 2:
        VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, *farneback_params)
        VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, *farneback_params)
      else:
        VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, None, *farneback_params)
        VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, None, *farneback_params)
      
      # For actual interpolation, original vectors V1 and V2 are used!
      R1_warped = cv2.remap(R1, W-tw*VF,       None, cv2.INTER_LINEAR)
      R2_warped = cv2.remap(R2, W-(1.0-tw)*VB, None, cv2.INTER_LINEAR)

      MASK1  = logical_and(isfinite(R1_warped), R1_warped != missingval)
      MASK2  = logical_and(isfinite(R2_warped), R2_warped != missingval)
      MASK12 = logical_and(MASK1, MASK2)

      R_interp_cur = ones(R1_warped.shape) * missingval
      R_interp_cur[MASK12] = (1.0-tw)*R1_warped[MASK12] + tw*R2_warped[MASK12]
      MASK_ = logical_and(MASK1, ~MASK2)
      R_interp_cur[MASK_] = R1_warped[MASK_]
      MASK_ = logical_and(MASK2, ~MASK1)
      R_interp_cur[MASK_] = R2_warped[MASK_]
      # Adding interpolated image to list
      R_interp[(int(((tws==tw).nonzero())[0])+1),:,:] = (R_interp_cur)

  # Masking in the previous step does not take into account that missing values can actually spread and dilute to actual field. Therefore appylying a simple check to remove spurious values from fields, based on absolute maximum/minimum values in either field.
  R_interp[R_interp>(R_max*1.02)] = missingval
  R_interp[R_interp<(R_min*0.98)] = missingval

  return R_interp


# This function smooths out the image and turns the data values to ubyte type for the motion vector calculation
def _filtered_ubyte_image(I, mask_nodata, R_min, R_max, filter_stddev=1.0, logtrans=False):
  I = I.copy()
  
  MASK = ~np.ma.getmask(mask_nodata)
  I[I > R_max] = R_max
  
  if logtrans == True:
    I = log(I)
    R_min = log(R_min)
    R_max = log(R_max)
  # Scale actual data points to have float values between 128...255. The missing values are assigned with the value of 0.
  I[MASK] = 128.0 + (I[MASK] - R_min) / (R_max - R_min) * 127.0
  I[~MASK] = 0.0
  
  I = I.astype(ubyte)
  I = gaussian_filter(I, sigma=filter_stddev)
  return I


# Sigmoid function
def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def logistic_growth_function(x):
    return 1 / (1 + np.exp(-0.5*x))
