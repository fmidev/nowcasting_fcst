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
import cv2
from eccodes import *
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
# import pygrib
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid

########### DEBUGGING FUNCTIONS #############

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

def plot_imshow_on_map(temps,vmin,vmax,outfile,cmap,title,longitudes,latitudes):
    # Transform longitudes defined in 0...360 to -180...180
    longitudes[longitudes>180] = (180-longitudes[(longitudes>180)]%180)
    grid_lon, grid_lat = [longitudes, latitudes]
    m = Basemap(projection='cyl', llcrnrlon=longitudes.min(),urcrnrlon=longitudes.max(),llcrnrlat=latitudes.min(),urcrnrlat=latitudes.max(),resolution='c')
    x, y = m(*[grid_lon, grid_lat])
    m.pcolormesh(x,y,temps,shading='flat',cmap=cmap,vmin=vmin,vmax=vmax)
    m.colorbar(location='right')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90.,120.,10.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1])
    
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








