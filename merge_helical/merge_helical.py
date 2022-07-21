''' Code to transform helical scan data to normal scan data readable by tomopy-cli
Original version written by Viktor Nikitin at 2-BM.

'''
from pathlib import Path
import numpy as np
import sys
import h5py
import numpy as np
import cupy as  cp # subpixel shifts on gpu
#import numpy as  cp # subpixel shifts on cpu
from merge_helical import handle_hdf, log, file_io, prep


def apply_shift_subpixel(data, shifts, pad=1):
    """Apply shifts for projections on GPU."""

    [ntheta, nz, n] = data.shape
    # padding
    tmp = cp.zeros([ntheta, nz+2*pad, n], dtype='float32')
    tmp[:, pad:-pad] = data
    # shift in the frequency domain
    y = cp.fft.fftfreq(nz+2*pad).astype('float32').reshape([nz+2*pad,1])        
    s = cp.exp(-2*np.pi*1j * (y*cp.array(shifts[:,  None, None])))   
    data = cp.fft.irfft2(s*np.fft.rfft2(tmp))
    return data


def compute_helical_params(params):
    '''Computes the pixel shift per projection and the number of output angles.

    Takes data from the meta data of the HDF5 file.
    '''
    with h5py.File(params.file_name, 'r') as hdf_file:
        scan_type = hdf_file['/process/acquisition/scan_type'][0].decode('UTF-8')
        log.info(f'scan type = {scan_type}')
        if scan_type.lower() != "helical":
            log.info("  not a helical scan, so nothing to do")
            return None
        pixels_per_360deg = hdf_file['/process/acquisition/pixels_y_per_360_deg'][0]
        theta = hdf_file['/exchange/theta'][...]
        flip_stitch = hdf_file['/process/acquisition/flip_stitch'][0].decode('UTF-8')
        if flip_stitch.lower() == 'yes':
            theta_max = theta[theta - theta[0] <= 360][-1]
            log.info(f'  flip and stitch scan, theta range {theta[0]} to {theta_max}')
        else:
            theta_max = theta[theta - theta[0] <= 180][-1]
            log.info(f'   0 - 180 degree data, theta range {theta[0]} to {theta_max}')
        data_size = hdf_file['/exchange/data'].shape
    params = file_io.auto_read_dxchange(params)
    if theta_max == theta[-1]:
        params.final_theta = theta
    else:
        params.final_theta = theta[0:np.argmin(np.abs(theta - theta_max)) + 1]
    params.final_shifts = (theta - theta[0]) / 360. * pixels_per_360deg 
    params.final_y_size = data_size[1] + 2 * params.subpixel_pad + np.abs(int(np.ceil(params.final_shifts[-1])))
    return params


def make_skeleton_hdf(fname, fname_out, params):
    '''Set up new HDF file.
    '''
    with h5py.File(fname,'r') as fid, h5py.File(fname_out,'w') as fid_out:        
        # copy h5 file
        filter_data = ['data','data_white','data_dark','theta'] # will not be copied
        handle_hdf.copy_h5(fid,fid_out,filter_data,log=True)        
                
        [ntheta,nz,n] = fid['/exchange/data'].shape
        data_out = fid_out.create_dataset('/exchange/data',
                                        [params.final_theta.size,params.final_y_size,n],
                                        dtype='float32',fillvalue=0)        

        # create resulting angles
        fid_out.create_dataset('/exchange/theta',data=params.final_theta)
        
        # create resulting flat and dark fields
        fid_out.create_dataset('/exchange/data_dark',data=np.zeros([1,params.final_y_size,n]),dtype='float32')
        fid_out.create_dataset('/exchange/data_white',data=np.ones([1,params.final_y_size,n]),dtype='float32')


def merge_helical(params): 
    
    fname = params.file_name
    ptheta = params.proj_chunk_size
    pad = params.subpixel_pad 
    params = compute_helical_params(params)
    if not params:
        return
    ny_out = params.final_y_size
    ntheta_out = params.final_theta.size
    fname_out = fname.parent.joinpath(fname.stem +'_merged.h5')
    make_skeleton_hdf(fname, fname_out, params)
    print(params)
    print(params.final_shifts[:10])
    print(params.final_shifts[-10:])
    #import pdb; pdb.set_trace()
    with h5py.File(fname,'r') as fid, h5py.File(fname_out,'r+') as fid_out:        
        data_out = fid_out['/exchange/data']

        # calculate shifts
        shifts = params.final_shifts
        [ntheta, ny, nx] = fid['/exchange/data'].shape

        sino = (0, ny)
        # shift data by chunks 
        for k in range(int(np.ceil(ntheta/ptheta))): 
            st = k * ptheta 
            end = min(ntheta,(k+1)*ptheta)
            print(f'Processing angle chunk {st}, {end}')
            proj, flat, dark, theta = file_io.read_tomo(sino, (st, end), params) 

            # Apply all preprocessing functions
            data = prep.all(proj, flat, dark, params, sino)
            del(proj, flat, dark)
            #import pdb; pdb.set_trace() 
            data_chunk = cp.array(data)

            # integer + float shifts
            ishifts = np.int32(shifts[st:end])
            fshifts = np.float32(shifts[st:end]-ishifts)
            
            if shifts[1]>shifts[0]:
                #stage is moving up
                stz = ishifts
                endz = ishifts + ny + 2 * pad
            else:                 
                #stage is moving down
                endz = ny_out - 1 + ishifts
                stz = ny_out - 1 + ishifts - ny - 2 * pad                
            data_chunk = apply_shift_subpixel(data_chunk, fshifts, pad)
            if not isinstance(data_chunk, np.ndarray):
                data_chunk = data_chunk.get()
            for kk in range(end-st):
                data_out[(kk+st)%ntheta_out, stz[kk]:endz[kk]] += data_chunk[kk]    
            
