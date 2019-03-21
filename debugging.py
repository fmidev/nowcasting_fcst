import interpolate_fcst
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

import PyQt4
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm



#Parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--obsdata',
                    default="testdata/2019020406/obs_2r.grib2",
                    help='Obs data, representing the first time step used in image morphing.')
parser.add_argument('--modeldata',
                    default="testdata/2019020406/fcst_2r.grib2",
                    help='Model data, from the analysis timestamp up until the end of the available 10-day forecast.')
parser.add_argument('--seconds_between_steps',
                    type=int,
                    default=3600,
                    help='Seconds between interpolated steps.')
parser.add_argument('--interpolated_data',
                    default='testdata/2019020406/output/interpolated_uusi2_2r.grib2',
                    help='Output file name for nowcast data.')
parser.add_argument('--predictability',
                    type=int,
                    default='4',
                    help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
parser.add_argument('--parameter',
                    default='Temperature',
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









# FUNCIONS USED

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



def read(image_file):
    if image_file.endswith(".nc"):
        return read_nc(image_file)
    elif image_file.endswith(".grib2"):
        return read_grib(image_file)
    elif image_file.endswith(".hdf"):
        return read_HDF5(image_file)
    else:
        print "unsupported file type for file: %s" % (image_file)




def read_grib(image_grib_file):

    # check comments from read_nc()
    dtime = []
    tempsl = []

    with GribFile(image_grib_file) as grib:
        for msg in grib:
            #print msg.size()

            ni = msg["Ni"]
            nj = msg["Nj"]

            forecast_time = datetime.datetime.strptime("%s%s" % (msg["dataDate"], msg["dataTime"]), "%Y%m%d%H%M") + datetime.timedelta(hours=msg["forecastTime"])
            dtime.append(forecast_time)
            tempsl.append(np.asarray(msg["values"]).reshape(nj, ni))

    temps = np.asarray(tempsl)
    nodata = 9999

    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    return temps, temps_min, temps_max, dtime, mask_nodata, nodata




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

    # The script returns four variables: the actual data, timestamps, nodata_mask and the actual nodata value
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

    # This edits each grib message individually (and assumes that forecastTime is an integer value)
    with GribFile(image_grib_file) as grib:
        for msg in grib:
            msg["generatingProcessIdentifier"] = 202
            msg["centre"] = 86
            msg["bitmapPresent"] = True
            i = msg["forecastTime"]
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
    config = ConfigParser.RawConfigParser()
    config.read(config_file_name)
    farneback_pyr_scale  = config.getfloat("optflow",    "pyr_scale")
    farneback_levels     = config.getint("optflow",      "levels")
    farneback_winsize    = config.getint("optflow",      "winsize")
    farneback_iterations = config.getint("optflow",      "iterations")
    farneback_poly_n     = config.getint("optflow",      "poly_n")
    farneback_poly_sigma = config.getfloat("optflow",    "poly_sigma")
    farneback_params = (farneback_pyr_scale,  farneback_levels, farneback_winsize, 
                        farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0)
    return farneback_params






















def plot_imshow(temps,vmin,vmax,outfile,cmap,title):
    plt.imshow(temps,cmap=cmap,vmin=vmin,vmax=vmax,origin="lower")
    #plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout(pad=0.)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_only_colorbar(vmin,vmax,units,outfile,cmap):
    fig = plt.figure(figsize=(8, 1))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1,cmap=cmap, norm=norm,orientation='horizontal')
    cb1.set_label(units)
    plt.savefig(outfile,bbox_inches='tight')
    plt.close()


def plot_verif_scores(fc_lengths,verif_scores,labels,outfile,title,y_ax_title):
    for n in range(0, verif_scores.shape[0]):
        plt.plot(fc_lengths, verif_scores[n,:], linewidth=2.0, label=str(labels[n]))
    plt.legend(bbox_to_anchor=(0.21, 1))
    plt.title(title)
    plt.xlabel('Forecast length (h)')
    plt.ylabel(y_ax_title)
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()















################# MAIN ##############



# #Read parameters from config file for interpolation (or optical flow algorithm, find out this later!). The function for reading the parameters is defined above.
farneback_params=farneback_params_config(options.farneback_params)

# In the "verification mode", the idea is to load in the "observational" and "forecast" datasets as numpy arrays. Both of these numpy arrays ("image_array") contain ALL the timesteps contained also in the files themselves. In addition, the returned variables "timestamp" and "mask_nodata" contain the values for all the timesteps.
# In "forecast mode", only one model timestep is given as an argument

# First precipitation field is from Tuliset2/analysis. What about possible scaling???
if options.parameter == 'Precipitation1h_TULISET': # Not needed atm
    image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1 = read_HDF5("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613120000.hdf")
else:
    image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1 = read(options.obsdata)
    quantity1 = options.parameter
# The second field is always the edited forecast field
image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2 = read(options.modeldata)
quantity2 = options.parameter

# In verification mode timestamps must be the same, otherwise exiting!
if (options.mode == "verif" and sum(timestamp1 == timestamp2)!=timestamp1.shape[0]):
   raise ValueError("obs and model data have different timestamps!")
   # sys.exit( "Timestamps do not match!" )    

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

interpolated_advection_inprogram = interpolated_advection

interpolated_advection, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1 = read(options.interpolated_data)


# NOW PLOT DIAGNOSTICS FROM THE FIELDS

# TIME SERIES FROM THE FIELD MEAN
fc_lengths=np.arange(0,6)
outdir = "figures/"
outfile = outdir + "Field_mean_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + ".png"
plt.plot(fc_lengths, np.mean(interpolated_advection,axis=(1,2)), linewidth=2.0, label="temperature")
title = "Field mean, " + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S")
plt.title(title)
plt.tight_layout(pad=0.)
# plt.xticks([])
# plt.yticks([])
plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
plt.close()

# JUMPINESS CHECKS
# PREVAILING ASSUMPTION: THE CHANGE IN INDIVIDUAL GRIDPOINTS IS VERY LINEAR

# RESULT ARRAYS
linear_change_boolean = np.ones(interpolated_advection.shape)
gp_abs_difference = np.ones(interpolated_advection.shape)
gp_mean_difference = np.ones(interpolated_advection.shape)
ratio_meandiff_absdiff = np.ones(interpolated_advection.shape)

# 0) PLOT DMO FIELDS
for n in (range(0, 6)):
    outdir = "figures/fields/"
    # PLOTTING AND SAVING TO FILE
    title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + "fc=+" + str(n) + "h"
    outfile = outdir + "field" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(interpolated_advection[n,:,:],R_min,R_max,outfile,"jet",title)

# 1a) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
for n in (range(1, 5)):
    outdir = "figures/linear_change/"
    gp_increased = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] > 0) & (interpolated_advection[(n+1),:,:] - interpolated_advection[n,:,:] > 0)
    gp_reduced = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] < 0) & (interpolated_advection[(n+1),:,:] - interpolated_advection[n,:,:] < 0)
    linear_change_boolean[n,:,:] = gp_reduced + gp_increased
    gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+1),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+1),:,:]))))))
    # PLOTTING AND SAVING TO FILE
    title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+1) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
    outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)

# 1b) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+1 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
for n in (range(1, 4)):
    outdir = "figures/linear_change3h/"
    gp_increased = ((interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:]) > 0) & ((interpolated_advection[(n+2),:,:] - interpolated_advection[n,:,:]) > 0)
    gp_reduced = ((interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:]) < 0) & ((interpolated_advection[(n+2),:,:] - interpolated_advection[n,:,:]) < 0)
    linear_change_boolean[n,:,:] = gp_reduced + gp_increased
    gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+2),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+2),:,:]))))))
    # PLOTTING AND SAVING TO FILE
    title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+2) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
    outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)

# 1c) TRUE IS ASSIGNED TO THOSE TIME+LOCATION POINTS THAT HAVE PREVIOUS TIMESTEP VALUE LESS (MORE) AND NEXT+2 TIMESTEP VALUE MORE (LESS) THAN THE CORRESPONDING VALUE. SO NO CHANGE IN THE SIGN OF THE DERIVATIVE.
for n in (range(1, 3)):
    outdir = "figures/linear_change4h/"
    gp_increased = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] > 0) & (interpolated_advection[(n+3),:,:] - interpolated_advection[n,:,:] > 0)
    gp_reduced = (interpolated_advection[n,:,:] - interpolated_advection[(n-1),:,:] < 0) & (interpolated_advection[(n+3),:,:] - interpolated_advection[n,:,:] < 0)
    linear_change_boolean[n,:,:] = gp_reduced + gp_increased
    gp_outside_minmax_range = np.max(np.maximum(np.maximum(0,(interpolated_advection[n,:,:]-np.maximum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+3),:,:]))),abs(np.minimum(0,(interpolated_advection[n,:,:]-np.minimum(interpolated_advection[(n-1),:,:],interpolated_advection[(n+3),:,:]))))))
    # PLOTTING AND SAVING TO FILE
    title = options.parameter + " " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n) + "h, " + str(round(np.mean(linear_change_boolean[n,:,:])*100,2)) + "% gridpoints is \n inside range " + str(n-1) + "h..." + str(n+3) + "h, outside range field max value " + str(round(gp_outside_minmax_range,2))
    outfile = outdir + "linear_change_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(linear_change_boolean[n,:,:],0,1,outfile,"jet",title)


# ANALYSING "JUMPINESS"
for n in (range(1, 5)):
    # 2) DIFFERENCE OF TIMESTEPS (n-1) AND (n+1)
    gp_abs_difference[n,:,:] = abs(interpolated_advection[(n+1),:,:] - interpolated_advection[(n-1),:,:])
    outdir = "figures/jumpiness_absdiff/"
    title = options.parameter + " absdiff of " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h \n field max value " + str(round(np.max(gp_abs_difference[n,:,:]),2))
    outfile = outdir + "jumpiness_absdiff_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(gp_abs_difference[n,:,:],-5,5,outfile,"seismic",title)
    # 3) DIFFERENCE FROM THE MEAN OF (n-1) AND (n+1)
    gp_mean_difference[n,:,:] = abs(interpolated_advection[n,:,:] - ((interpolated_advection[(n+1),:,:] + interpolated_advection[(n-1),:,:])/2))
    outdir = "figures/jumpiness_meandiff/"
    title = options.parameter + " diff from " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h_mean \n field max value " + str(round(np.max(gp_mean_difference[n,:,:]),2))
    outfile = outdir + "jumpiness_meandiff_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(gp_mean_difference[n,:,:],-0.2,0.2,outfile,"seismic",title)
    # 4) MEAN DIFF DIVIDED BY ABSDIFF
    ratio_meandiff_absdiff = gp_mean_difference / gp_abs_difference
    outdir = "figures/jumpiness_ratio/"
    title = options.parameter + " meandiff / absdiff ratio % " + timestamp1[0].strftime("%Y%m%d%H%M%S") + " fc_" + str(n-1) + "h" + str(n+1) + "h \n field max value " + str(round(np.max(ratio_meandiff_absdiff[n,:,:]),2))
    outfile = outdir + "jumpiness_ratio_" + options.parameter + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_fc=+" + str(n) + "h.png"
    plot_imshow(ratio_meandiff_absdiff[n,:,:],0,2,outfile,"seismic",title)

# Save interpolated field to a new nc file
write(interpolated_data=interpolated_advection,image_file=options.modeldata,write_file=options.interpolated_data,variable=options.parameter,predictability=options.predictability)
