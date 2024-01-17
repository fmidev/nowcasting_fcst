import configparser
import numpy as np

def farneback_params_config(config_file_name):
    config = configparser.ConfigParser()
    config.read(config_file_name)
    farneback_pyr_scale = config.getfloat("optflow", "pyr_scale")
    farneback_levels = config.getint("optflow", "levels")
    farneback_winsize = config.getint("optflow", "winsize")
    farneback_iterations = config.getint("optflow", "iterations")
    farneback_poly_n = config.getint("optflow", "poly_n")
    farneback_poly_sigma = config.getfloat("optflow", "poly_sigma")
    farneback_params = (
        farneback_pyr_scale,
        farneback_levels,
        farneback_winsize,
        farneback_iterations,
        farneback_poly_n,
        farneback_poly_sigma,
        0,
    )
    return farneback_params

def define_common_mask_for_fields(*args):
    """Calculate a combined mask for each input. Some input values might have several timesteps, but here define a mask if ANY timestep for that particular gridpoint has a missing value"""
    stacked = np.sum(np.ma.getmaskarray(args[0]), axis=0) > 0
    if len(args) == 0:
        return stacked
    for arg in args[1:]:
        try:
            stacked = np.logical_or(
                stacked, (np.sum(np.ma.getmaskarray(arg), axis=0) > 0)
            )
        except:
            raise ValueError("grid sizes do not match!")
    return stacked

