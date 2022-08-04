import numpy as np
import xarray as xr
import pygrib as pg
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy


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
    # For those longitudes that are over 180 degrees, reduce 360 degrees from them
    longitudes[longitudes > 180] = longitudes[longitudes > 180] - 360

    grid_lon, grid_lat = [longitudes, latitudes]
    
    proj = cartopy.crs.LambertConformal(central_latitude = int(np.mean(latitudes)), 
                             central_longitude = int(np.mean(longitudes)), 
                             standard_parallels = (25, 25))
    #proj = cartopy.crs.PlateCarree()
    #ncolors = 256
    
    temps[np.where(temps<=0.1)] = None
    ncolors = 128
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
    # create a colormap object
    # register this new colormap with matplotlib
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)
    ax = plt.axes(projection = proj)
    # x, y = ax(*[grid_lon, grid_lat])
    x, y = [grid_lon, grid_lat]
    cm = ax.pcolormesh(x, y, temps, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap, transform=cartopy.crs.PlateCarree())
    ax.coastlines('50m')
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)
    ax.gridlines()
    ax.set_title(title)
    plt.colorbar(cm, fraction = 0.046, pad = 0.04, orientation="horizontal")
    # plt.show()

    # # Transform longitudes defined in 0...360 to -180...180
    # longitudes[longitudes>180] = (180-longitudes[(longitudes>180)]%180)
    # grid_lon, grid_lat = [longitudes, latitudes]
    # m = Basemap(projection='cyl', llcrnrlon=longitudes.min(),urcrnrlon=longitudes.max(),llcrnrlat=latitudes.min(),urcrnrlat=latitudes.max(),resolution='c')
    # x, y = m(*[grid_lon, grid_lat])
    # m.pcolormesh(x,y,temps,shading='flat',cmap=cmap,vmin=vmin,vmax=vmax)
    # m.colorbar(location='right')
    # m.drawcoastlines()
    # m.drawmapboundary()
    # m.drawparallels(np.arange(-90.,120.,10.),labels=[1,0,0,0])
    # m.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1])
    
    # plt.title(title)
    # plt.tight_layout(pad=0.)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0, dpi=1200)
    plt.close()


def plot_contourf_map_scandinavia(grib_file, vmin, vmax, outfile, date, title):
    ds = xr.load_dataset(grib_file)
    for v in ds:
        data = ds[v].data
        lats, lons = ds['latitude'].data, ds['longitude'].data
        lons[lons > 180] = lons[lons > 180] - 360
        proj = cartopy.crs.LambertConformal(central_latitude=int(np.mean(lats)),
                                            central_longitude=int(np.mean(lons)),
                                            standard_parallels=(25, 25))
        for i in range(len(data)):
            ax = plt.axes(projection=proj)
            ax.set_extent([5, 35, 52, 72])
            ax.gridlines()
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS)
            ax.add_feature(cartopy.feature.OCEAN)
            ax.add_feature(cartopy.feature.LAND)
            ax.add_feature(cartopy.feature.LAKES)
            ax.add_feature(cartopy.feature.RIVERS)
            d = data[i]
            d[d <= 0.00001] = np.nan
            cm = ax.pcolormesh(lons, lats, d, transform=cartopy.crs.PlateCarree(), shading='auto', vmin=vmin, vmax=vmax, cmap='OrRd')
            plt.title(f"Propability of Thunder, {date} forecast {i}h")
            plt.colorbar(cm, fraction=0.046, pad=0.04, orientation="horizontal", ax=ax)
            idx = outfile.index("h.")
            forecast_outfile = outfile[:idx] + f"{i}" + outfile[idx:]
            plt.savefig(forecast_outfile, bbox_inches='tight', pad_inches=0.2, dpi=800)
            plt.close()
            lol


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

