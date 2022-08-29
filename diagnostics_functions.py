import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import cartopy


def plot_imshow_map_scandinavia(grib_file, vmin, vmax, outfile, date, title):
    """Use for plotting when projection is Polster/Polar_stereografic

    Only for Scandinavian domain.
    For xarray to work with grib-files, cfgrib must be installed
    """
    ds = xr.load_dataset(grib_file)
    cmap = 'RdYlGn_r' # RdBl_r  'Blues' 'Jet' 'RdYlGn_r'
    for v in ds:
        data = ds[v].data
        lat_ts, lat0, lon0 = 52, 63, 19
        for i in range(len(data)):
            m = Basemap(width=1900000, height=2100000,
                        resolution='l', projection='laea',
                        lat_ts=lat_ts, lat_0=lat0, lon_0=lon0)
            m.drawcountries(linewidth=1.0)
            m.drawcoastlines(1.0)

            d = data[i]
            d[d <= 0.01] = np.nan
            cm = m.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", zorder=1)
            plt.title(f"{title}, {date} forecast {i}h")
            plt.colorbar(cm, fraction=0.046, pad=0.04, orientation="horizontal")
            idx = outfile.index("h.")
            forecast_outfile = outfile[:idx] + f"{i}" + outfile[idx:]
            plt.savefig(forecast_outfile, bbox_inches='tight', pad_inches=0.2, dpi=800)
            plt.close()


def plot_contourf_map_scandinavia(grib_file, vmin, vmax, outfile, date, title):
    """Use for plotting when projection is Lambert etc.

    For xarray to work with grib-files, cfgrib must be installed
    """
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
            cm = ax.pcolormesh(lons, lats, d, transform=cartopy.crs.PlateCarree(),
                               shading='auto', vmin=vmin, vmax=vmax, cmap='OrRd')
            plt.title(f"{title}, {date} forecast +{i}h")
            plt.colorbar(cm, fraction=0.046, pad=0.04, orientation="horizontal", ax=ax)
            idx = outfile.index("h.")
            forecast_outfile = outfile[:idx] + f"{i}" + outfile[idx:]
            plt.savefig(forecast_outfile, bbox_inches='tight', pad_inches=0.2, dpi=800)
            plt.close()


########### DEBUGGING FUNCTIONS #############
def plot_imshow(temps,vmin,vmax,outfile,cmap,title):
    plt.imshow(temps,cmap=cmap,vmin=vmin,vmax=vmax,origin="lower")
    #plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout(pad=0.)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()
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
