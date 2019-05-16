# -*- coding: utf-8 -*-
import interpolate_fcst
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser
import netCDF4
import sys
import os
import time
from eccodes import *
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt


# FUNCTIONS USED

def read(image_file):
    if image_file.endswith(".nc"):
        return read_nc(image_file)
    elif image_file.endswith(".grib2"):
        return read_grib(image_file)
    elif image_file.endswith(".h5"):
        return read_HDF5(image_file)
    else:
        print "unsupported file type for file: %s" % (image_file)
        


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim 
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars



def read_nc(image_nc_file):
    tempds = netCDF4.Dataset(image_nc_file)
    internal_variable = tempds.variables.keys()[-1]
    temps = np.array(tempds.variables[internal_variable][:]) # This picks the actual data
    nodata = tempds.variables[internal_variable].missing_value
    time_var = tempds.variables["time"]
    dtime = netCDF4.num2date(time_var[:],time_var.units) # This produces an array of datetime.datetime values
    
    # Outside of area all the values are missing. Leave them as they are. They're not affected by the motion vector calculations
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    # Pick min/max values from the data
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata



def read_grib(image_grib_file):

    # check comments from read_nc()
    dtime = []
    tempsl = []
    latitudes = []
    longitudes = []
    
    with GribFile(image_grib_file) as grib:
        for msg in grib:
            #print msg.size()
            #print msg.keys()
            ni = msg["Ni"]
            nj = msg["Nj"]

            forecast_time = datetime.datetime.strptime("%s%02d" % (msg["dataDate"], msg["dataTime"]), "%Y%m%d%H%M") + datetime.timedelta(hours=msg["forecastTime"])
            dtime.append(forecast_time)
            tempsl.append(np.asarray(msg["values"]).reshape(nj, ni))
            latitudes.append(np.asarray(msg["latitudes"]).reshape(nj, ni))
            longitudes.append(np.asarray(msg["longitudes"]).reshape(nj, ni))

            
    temps = np.asarray(tempsl)
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    latitudes = latitudes[0,:,:]
    longitudes = longitudes[0,:,:]
    nodata = 9999

    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata, longitudes, latitudes



def read_HDF5(image_h5_file):
    #Read RATE or DBZH from hdf5 file
    print 'Extracting data from image h5 file'
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    # #Read RATE array if found in dataset      
    # test=comp.select_dataset('RATE')
    # if test != None:
    #     image_array=comp.dataset
    #     quantity='RATE'
    # else:
    #     #Look for DBZH array
    #     test=comp.select_dataset('DBZH')
    #     if test != None:
    #         image_array=comp.dataset
    #         quantity='DBZH'
    #     else:
    #         print 'Error: RATE or DBZH array not found in the input image file!'
    #         sys.exit(1)
    #Look for DBZH array
    test=comp.select_dataset('DBZH')
    if test != None:
        image_array=comp.dataset
        quantity='DBZH'
    else:
        print 'Error: RATE or DBZH array not found in the input image file!'
        sys.exit(1)
    if quantity == 'RATE':
        quantity_min = options.R_min
        quantity_max = options.R_max
    if quantity == 'DBZH':
        quantity_min = options.DBZH_min
        quantity_max = options.DBZH_max
    
    #Read nodata and undetect values from metadata for masking
    gen = comp.attr_gen('nodata')
    pair = gen.next()
    nodata = pair.value
    gen = comp.attr_gen('undetect')
    pair = gen.next()
    undetect = pair.value
    #Read gain and offset values from metadata
    gen = comp.attr_gen('gain')
    pair = gen.next()
    gain = pair.value
    gen = comp.attr_gen('offset')
    pair = gen.next()
    offset = pair.value
    #Read timestamp from metadata
    gen = comp.attr_gen('date')
    pair = gen.next()
    date = pair.value
    gen = comp.attr_gen('time')
    pair = gen.next()
    time = pair.value

    timestamp=date+time
    dtime = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")

    # Flip latitude dimension
    image_array = np.flipud(image_array)
    tempsl = []
    tempsl.append(image_array)
    temps = np.asarray(tempsl)
    
    #Masks of nodata and undetect
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    mask_undetect = np.ma.masked_where(temps == undetect,temps)
    #Change to physical values
    temps=temps*gain+offset
    #Mask undetect values as 0 and nodata values as nodata
    temps[np.where(np.ma.getmask(mask_undetect))] = 0
    temps[np.where(np.ma.getmask(mask_nodata))] = nodata
    
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata



def write(interpolated_data,image_file,write_file,variable,predictability):

    if write_file.endswith(".nc"):
        write_nc(interpolated_data,image_file,write_file,variable,predictability)
    elif write_file.endswith(".grib2"):
        write_grib(interpolated_data,image_file,write_file,variable,predictability)
    else:
        print "unsupported file type for file: %s" % (image_file)
        return
    print "wrote file '%s'" % write_file



def write_grib(interpolated_data,image_grib_file,write_grib_file,variable,predictability):
    # (Almost) all the metadata is copied from modeldata.grib2
    try:
        os.remove(write_grib_file)
    except OSError,e:
        pass

    # Change data type of numpy array
    # interpolated_data = np.round(interpolated_data,2)
    # interpolated_data = interpolated_data.astype('float64')             

    #with GribFile(image_grib_file) as grib:
    #    for msg in grib:
    #        msg["generatingProcessIdentifier"] = 202
    #        msg["centre"] = 86
    #        msg["bitmapPresent"] = True
    #
    #        for i in range(interpolated_data.shape[0]):
    #            msg["forecastTime"] = i
    #            msg["values"] = interpolated_data[i].flatten()
    #
    #            with open(write_grib_file, "a") as out:
    #                msg.write(out)
    #        break # we use only the first grib message as a template

    # This edits each grib message individually
    with GribFile(image_grib_file) as grib:
        i=-1
        for msg in grib:
            msg["generatingProcessIdentifier"] = 202
            msg["centre"] = 86
            msg["bitmapPresent"] = True
            i = i+1 # msg["forecastTime"]
            if (i == interpolated_data.shape[0]):
                break
            msg["values"] = interpolated_data[i,:,:].flatten()
            # print("{} {}: {}".format("histogram of interpolated_data timestep ",i,np.histogram(interpolated_data[i,:,:].flatten(),bins=20,range=(240,300))))
            # print("{} {}: {}".format("histogram of msg[values] timestep ",i,np.histogram(msg["values"],bins=20,range=(240,300))))
            with open(write_grib_file, "a") as out:
                msg.write(out)



def write_nc(interpolated_data,image_nc_file,write_nc_file,variable,predictability):
    
    # All the metadata is copied from modeldata.nc
    nc_f = image_nc_file
    nc_fid = netCDF4.Dataset(nc_f, 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)
    # Open the output file
    w_nc_fid = netCDF4.Dataset(write_nc_file, 'w', format='NETCDF4')
    # Using our previous dimension information, we can create the new dimensions
    data = {}
    # Creating dimensions
    for dim in nc_dims:
        w_nc_fid.createDimension(dim, nc_fid.variables[dim].size)
    # Creating variables
    for var in nc_vars:
        data[var] = w_nc_fid.createVariable(varname=var, datatype=nc_fid.variables[var].dtype,dimensions=(nc_fid.variables[var].dimensions))
        # Assigning attributes for the variables
        for ncattr in nc_fid.variables[var].ncattrs():
            data[var].setncattr(ncattr, nc_fid.variables[var].getncattr(ncattr))
        # Assign the data itself
        if (var in ['time','y','x','lat','lon']):
            w_nc_fid.variables[var][:] = nc_fid.variables[var][:]
        if (var == nc_vars[len(nc_vars)-1]):
            w_nc_fid.variables[var][:] = interpolated_data
    # Creating the global attributes
    w_nc_fid.description = "Blended nowcast forecast, variable %s, predictability %s" %(variable, predictability)
    w_nc_fid.history = 'Created ' + time.ctime(time.time())  
    for attribute in nc_attrs:
        w_nc_fid.setncattr(attribute, nc_fid.getncattr(attribute))
        # w_nc_fid[attribute] = nc_fid.getncattr(attribute)
    # for attribute in nc_attrs:
    #     eval_string = 'w_nc_fid.' + attribute + ' = nc_fid.getncattr("' + attribute + '")'
    #     eval(eval_string)
    # Close the file
    w_nc_fid.close()



def farneback_params_config(config_file_name):
    config = ConfigParser.ConfigParser()
    config.read(config_file_name)
    farneback_pyr_scale  = config.getfloat("optflow", "pyr_scale")
    farneback_levels     = config.getint("optflow", "levels")
    farneback_winsize    = config.getint("optflow", "winsize")
    farneback_iterations = config.getint("optflow", "iterations")
    farneback_poly_n     = config.getint("optflow", "poly_n")
    farneback_poly_sigma = config.getfloat("optflow", "poly_sigma")
    farneback_params = (farneback_pyr_scale,  farneback_levels, farneback_winsize, 
                        farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0)
    return farneback_params



def main():
    
    # #Read parameters from config file for interpolation or optical flow algorithm. The function for reading the parameters is defined above.
    farneback_params=farneback_params_config(options.farneback_params)

    # In the "verification mode", the idea is to load in the "observational" and "forecast" datasets as numpy arrays. Both of these numpy arrays ("image_array") contain ALL the timesteps contained also in the files themselves. In addition, the returned variables "timestamp" and "mask_nodata" contain the values for all the timesteps.
    # In "forecast mode", only one model timestep is given as an argument

    # Always load in at least observation data and model data
    image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1, longitudes1, latitudes1 = read(options.obsdata)
    quantity1 = options.parameter
    # The second field is always the forecast field to where the blending is done to
    image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2, longitudes2, latitudes2 = read(options.modeldata)
    quantity2 = options.parameter

    # In verification mode timestamps must be the same, otherwise exiting!
    if (options.mode == "verif" and sum(timestamp1 == timestamp2)!=timestamp1.shape[0]):
        raise ValueError("obs and model data have different timestamps!")
        # sys.exit( "Timestamps do not match!" )    

    # If the the smoothed parameter is NOT precipitation_bg_1h
    if options.parameter != 'precipitation_bg_1h':
        # Defining a masked array that is the same for all forecast fields and both producers (if even one forecast length is missing for a specific grid point, that grid point is masked out from all fields).
        mask_nodata1_p = np.sum(np.ma.getmask(mask_nodata1),axis=0)>0
        mask_nodata2_p = np.sum(np.ma.getmask(mask_nodata2),axis=0)>0
        mask_nodata = np.logical_or(mask_nodata1_p,mask_nodata2_p)
        mask_nodata = np.ma.masked_where(mask_nodata == True,mask_nodata)
        del mask_nodata1_p,mask_nodata2_p

        # Missing data values are taken from the first field.
        nodata = nodata2 = nodata1

        # From both matrices, replace all values according to mask_nodata
        image_array1[:,mask_nodata] = nodata
        image_array2[:,mask_nodata] = nodata

        # Defining definite min/max values from the two fields
        R_min=min(quantity1_min,quantity2_min)
        R_max=max(quantity1_max,quantity2_max)

        # Checking if all latitude/longitude/timestamps in the different data sources correspond to each other
        if (np.array_equal(longitudes1,longitudes2)):
            longitudes = longitudes1
        if (np.array_equal(latitudes1,latitudes2)):
            latitudes = latitudes1

    else:
        # Read in also background field in to where Finnish data is merged
        image_array3, quantity3_min, quantity3_max, timestamp3, mask_nodata3, nodata3, longitudes3, latitudes3 = read(options.bgdata)
        quantity3 = options.parameter

        # Read radar composite field used as mask for LAPS. Lat/lon info is not stored in radar composite HDF5 files, but these are the same! (see README.txt)
        image_array4, quantity4_min, quantity4_max, timestamp4, mask_nodata4, nodata4 = read(options.detectabilitydata)
        # As the detectability field is used only as a mask, change all not-nodata values to zero
        image_array4[np.where(~np.ma.getmask(mask_nodata4))] = 0

        # Checking if all latitude/longitude/timestamps in the different data sources correspond to each other
        if (np.array_equal(longitudes1,longitudes2) and np.array_equal(longitudes1,longitudes3) and np.array_equal(latitudes1,latitudes2) and np.array_equal(latitudes1,latitudes3)):
            longitudes = longitudes1
            latitudes = latitudes1

            # Defining a masked array that is the same for all forecast fields and both producers (if even one forecast length is missing for a specific grid point, that grid point is masked out from all three fields).
            mask_nodata1_p = np.sum(np.ma.getmask(mask_nodata1),axis=0)>0
            mask_nodata2_p = np.sum(np.ma.getmask(mask_nodata2),axis=0)>0
            mask_nodata3_p = np.sum(np.ma.getmask(mask_nodata3),axis=0)>0
            mask_nodata4_p = np.sum(np.ma.getmask(mask_nodata4),axis=0)>0
            # Combining masks from radar detectability and LAPS
            mask_nodata5_p = np.sum(np.ma.getmask(mask_nodata4)+np.ma.getmask(mask_nodata1),axis=0)>0

        
            # Creating a linear smoother field: More weight for bg near the bg/obs border and less at the center of obs field
            # Gaussian smoother widens the coefficients so initially calculate from smaller mask the values
            mask_smaller_than_borders = 4 # Controls how many pixels the initial mask is compared to obs field
            smoother_coefficient = 0.2 # Controls how long from the borderline bg field is affecting
            gaussian_filter_coefficient = 3 # Sets the smoothiness of the weight field
            used_mask = distance_transform_edt(np.logical_not(mask_nodata5_p))
            used_mask = np.where(used_mask <= mask_smaller_than_borders,True,False)
            used_mask = distance_transform_edt(np.logical_not(used_mask))
            weights_obs = gaussian_filter(used_mask,gaussian_filter_coefficient)
            weights_obs = weights_obs / (smoother_coefficient*np.max(weights_obs))
            # Cropping values to between 0 and 1
            weights_obs = np.where(weights_obs>1,1,weights_obs)
            weights_obs = np.where(np.logical_not(mask_nodata1_p),weights_obs,0)
            weights_bg = 1 - weights_obs

            # Adding up the two fields according to the weights and replacing the mask_nodata
            image_array1[0,:,:] = weights_obs*image_array1[0,:,:]+weights_bg*image_array3[0,:,:]
            mask_nodata1_p = mask_nodata3_p
            mask_nodata = np.logical_or(mask_nodata1_p,mask_nodata2_p)
            mask_nodata = np.ma.masked_where(mask_nodata == True,mask_nodata)
            del mask_nodata1_p,mask_nodata2_p,mask_nodata3_p
            # Missing data values are taken from the first field.
            nodata = nodata3 = nodata2 = nodata1
            # From both matrices, replace all values according to mask_nodata
            image_array1[:,mask_nodata] = nodata
            image_array2[:,mask_nodata] = nodata

            
            # Defining definite min/max values from the three fields
            R_min=min(quantity1_min,quantity2_min,quantity3_min)
            R_max=max(quantity1_max,quantity2_max,quantity3_max)
        else:
            print("latitudes/longitudes in some of the input files do not correspond to each other!")
            exit()
        
    # DATA IS NOW LOADED AS NORMAL NUMPY NDARRAYS

    #Resize observation field to same resolution with model field and slightly blur to make the two fields look more similar for OpenCV.
    # NOW THE PARAMETER IS A CONSTANT AD-HOC VALUE!!!
#     reshaped_size = list(image_array2.shape)
#     if (options.mode == "fcst"):
#         reshaped_size[0] = 1
#     image_array1_reshaped=np.zeros(reshaped_size)
#     for n in range(0,image_array1.shape[0]):
#         #Resize                       
#         image_array1_reshaped[n]=imresize(image_array1[n], image_array2[0].shape, interp='bilinear', mode='F')
#         #Blur
#         image_array1_reshaped[n]=gaussian_filter(image_array1_reshaped[n], options.gaussian_filter_sigma)
#     image_array1=image_array1_reshaped

    # CALCULATING INTERPOLATED IMAGES FOR DIFFERENT PRODUCERS AND CALCULATING VERIF METRICS

    # Like mentioned at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html, a reasonable value for poly_sigma depends on poly_n. Here we use a fraction 4.6 for these values.
    fb_params = (farneback_params[0],farneback_params[1],farneback_params[2],farneback_params[3],farneback_params[4],(farneback_params[4] / 4.6),farneback_params[6])
    # Interpolated data
    interpolated_advection=interpolate_fcst.advection(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, farneback_params=fb_params, predictability=options.predictability, seconds_between_steps=options.seconds_between_steps, R_min=R_min, R_max=R_max, missingval=nodata, logtrans=False)

    # Save interpolated field to a new file
    write(interpolated_data=interpolated_advection,image_file=options.modeldata,write_file=options.interpolated_data,variable=options.parameter,predictability=options.predictability)
    
    


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--obsdata',
                        default="testdata/2019032708/obs_tprate.grib2",
                        help='Obs data, representing the first time step used in image morphing.')
    parser.add_argument('--modeldata',
                        default="testdata/2019032708/fcst_tprate.grib2",
                        help='Model data, from the analysis timestamp up until the end of the available 10-day forecast.')
    parser.add_argument('--bgdata',
                        default="testdata/2019032708/mnwc_tprate.grib2",
                        help='Background field data where obsdata is merged to, having the same timestamp as obs data.')
    parser.add_argument('--detectabilitydata',
                        default="testdata/radar_detectability_field.h5",
                        help='Radar detectability field, which is used in spatial blending of obsdata and bgdata.')
    parser.add_argument('--seconds_between_steps',
                        type=int,
                        default=3600,
                        help='Seconds between interpolated steps.')
    parser.add_argument('--interpolated_data',
                        default='testdata/2019032708/output/interpolated_tprate.grib2',
                        help='Output file name for nowcast data.')
    parser.add_argument('--predictability',
                        type=int,
                        default='4',
                        help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
    parser.add_argument('--parameter',
                        default='temperature',
                        help='Variable which is handled.')
    parser.add_argument('--mode',
                        default='fcst',
                        help='Either "verif" or "fcst" mode. In verification mode, verification statistics are calculated from the blended forecasts. In forecast mode no.')
    parser.add_argument('--gaussian_filter_sigma',
                        type=float,
                        default=0.5,
                        help='This parameter sets the blurring intensity of the of the analysis field.')
    parser.add_argument('--R_min',
                        type=float,
                        default=0.1,
                        help='Minimum precipitation intensity for optical flow computations. Values below R_min are set to zero.')
    parser.add_argument('--R_max',
                        type=float,
                        default=30.0,
                        help='Maximum precipitation intensity for optical flow computations. Values above R_max are clamped.')
    parser.add_argument('--DBZH_min',
                        type=float,
                        default=10,
                        help='Minimum DBZH for optical flow computations. Values below DBZH_min are clamped.')
    parser.add_argument('--DBZH_max', 
                        type=float,
                        default=45,
                        help='Maximum DBZH for optical flow computations. Values above DBZH_max are clamped.')
    parser.add_argument('--farneback_params',
                        default='compute_advinterp.cfg',
                        help='location of farneback params configuration file')

    options = parser.parse_args()
    main()
