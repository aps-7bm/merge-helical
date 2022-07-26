import os
import logging

import tomopy
import dxchange
import numpy as np

from merge_helical import file_io
from merge_helical import beamhardening
from merge_helical import config

__all__ = ['all', 'remove_nan_neg_inf', 'cap_sinogram_values', 'zinger_removal', 'flat_correction', 
           'remove_stripe', 'phase_retrieval', 'minus_log', 'beamhardening_correct']


log = logging.getLogger(__name__)


def all(proj, flat, dark, params, sino):
    # zinger_removal
    proj, flat = zinger_removal(proj, flat, params)
    if (params.dark_zero):
        dark *= 0
        log.warning('  *** *** dark fields are ignored')

    # normalize
    data = flat_correction(proj, flat, dark, params)
    del(proj, flat, dark)
    # Perform beam hardening.  This leaves the data in pathlength.
    if params.beam_hardening_method == 'standard':
        data[:,...] = beamhardening_correct(data, params, sino)
    else:
        # minus log
        data = minus_log(data, params)
    # remove outlier
    data = remove_nan_neg_inf(data, params)
    data = cap_sinogram_values(data, params)
    return data


def remove_nan_neg_inf(data, params):

    log.info('  *** remove nan, neg and inf')
    if(params.fix_nan_and_inf == True):
        log.info('  *** *** ON')
        log.info('  *** *** replacement value %f ' % params.fix_nan_and_inf_value)
        data = tomopy.remove_nan(data, val=params.fix_nan_and_inf_value)
        data = tomopy.remove_neg(data, val= 0.0)
        data[np.isinf(data)] = params.fix_nan_and_inf_value
    else:
        log.warning('  *** *** OFF')
    return data


def cap_sinogram_values(data, params):
    log.info('  *** cap sinogram max value: %f', params.sinogram_max_value)
    data[data > params.sinogram_max_value] = params.sinogram_max_value
    return data


def zinger_removal(proj, flat, params):

    log.info("  *** zinger removal")
    if (params.zinger_removal_method == 'standard'):
        log.info('  *** *** ON')
        log.info("  *** *** zinger level projections: %d" % params.zinger_level_projections)
        log.info("  *** *** zinger level white: %s" % params.zinger_level_white)
        log.info("  *** *** zinger_size: %d" % params.zinger_size)
        proj = tomopy.misc.corr.remove_outlier(proj, params.zinger_level_projections, size=params.zinger_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(flat, params.zinger_level_white, size=params.zinger_size, axis=0)
    elif(params.zinger_removal_method == 'none'):
        log.warning('  *** *** OFF')

    return proj, flat


def flat_correction(proj, flat, dark, params):

    log.info('  *** normalization')
    if(params.flat_correction_method == 'standard'):
        try:
            data = tomopy.normalize(proj, flat, dark, 
                                cutoff=params.normalization_cutoff / params.bright_exp_ratio)
            data *= params.bright_exp_ratio
        except AttributeError:
            log.warning('  *** *** No bright_exp_ratio found.  Ignore')
        log.info('  *** *** ON %f cut-off' % params.normalization_cutoff)
    elif(params.flat_correction_method == 'air'):
        data = tomopy.normalize_bg(proj, air=params.air)
        log.info('  *** *** air %d pixels' % params.air)
    elif(params.flat_correction_method == 'none'):
        data = proj
        log.warning('  *** *** normalization is turned off')
    else:
        raise ValueError("Unknown value for *flat_correction_method*: {}. "
                         "Valid options are {}"
                         "".format(params.flat_correction_method,
                                   config.SECTIONS['flat-correction']['flat-correction-method']['choices']))
    return data


def minus_log(data, params):

    log.info("  *** minus log")
    if(params.minus_log):
        log.info('  *** *** ON')
        data = tomopy.minus_log(data)
    else:
        log.warning('  *** *** OFF')

    return data

def beamhardening_correct(data, params, sino):
    """
    Performs beam hardening corrections.
    Inputs
    data: data normalized already for bright and dark corrections.
    params: processing parameters
    sino: row numbers for these data
    """
    log.info("  *** correct beam hardening")
    data_dtype = data.dtype
    # Correct for centerline of fan
    softener = beamhardening.BeamSoftener(params)
    data = softener.fcorrect_as_pathlength_centerline(data)
    # Make an array of correction factors
    softener.center_row = params.center_row
    log.info("  *** *** Beam hardening center row = {:f}".format(softener.center_row))
    angles = np.abs(np.arange(sino[0], sino[1])- softener.center_row).astype(data_dtype)
    angles *= softener.pixel_size / softener.d_source
    log.info("  *** *** angles from {0:f} to {1:f} urad".format(angles[0], angles[-1]))
    correction_factor = softener.angular_spline(angles).astype(data_dtype)
    if len(data.shape) == 2:
        return data* correction_factor[:,None]
    else:
        return data * correction_factor[None, :, None]

