import os
import sys
import shutil
import h5py
from pathlib import Path
import argparse
import configparser
import numpy as np

from collections import OrderedDict

from merge_helical import log, util

LOGS_HOME = os.path.join(str(Path.home()), 'logs')
CONFIG_FILE_NAME = os.path.join(str(Path.home()), 'merge_helical.conf')
bh_data_path = Path(__file__).parent.joinpath('beam_hardening_data')

SECTIONS = OrderedDict()


SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration file",
        'metavar': 'FILE'},
    'logs-home': {
        'default': LOGS_HOME,
        'type': str,
        'help': "Log file directory",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'config-update': {
        'default': False,
        'help': 'When set, the content of the config file is updated using the current params values',
        'action': 'store_true'},
        }


SECTIONS['helical'] = {
    'pixels_per_360deg_auto': {
        'default': True,
        'help': 'Read pixels per 360deg from dxfile?',
        'action': 'store_true'},
    'pixels_per_360deg': {
        'default': 0.0,
        'type': float,
        'help': 'Vertical shift in pixels per 360deg',},
    'proj-chunk-size': {
        'default': 32,
        'type': int,
        'help': 'Number of projection angles to calculate at one time.',}, 
    'subpixel-pad': {
        'default': 1,
        'type': int,
        'help': 'Number of rows to pad when doing subpixel shifts.'},
        }


SECTIONS['file-reading'] = {
    'file-name': {
        'default': '.',
        'type': Path,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'file-format': {
        'default': 'dx',
        'type': str,
        'help': "see from https://dxchange.readthedocs.io/en/latest/source/demo.html",
        'choices': ['dx', 'anka', 'australian', 'als', 'elettra', 'esrf', 'aps1id', 'aps2bm', 'aps5bm', 'aps7bm', 'aps8bm', 'aps13bm', 'aps32id', 'petraP05', 'tomcat', 'xradia']},
    'file-type': {
        'default': 'standard',
        'type': str,
        'help': "Input file type",
        'choices': ['standard', 'flip_and_stich', 'double_fov', 'mosaic']},
    'binning': {
        'type': util.positive_int,
        'default': 0,
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': [0, 1, 2, 3]},
    'dark-zero': {
        'default': False,
        'help': 'When set, the the dark field is set to zero',
        'action': 'store_true'},
    'scintillator-auto': {
        'default': False,
        'help': "When set, read scintillator properties from the HDF file",
        'action': 'store_true'},
    'pixel-size-auto': {
        'default': False,
        'help': "When set, read effective pixel size from the HDF file",
        'action': 'store_true'},
       }

SECTIONS['zinger-removal'] = {
    'zinger-removal-method': {
        'default': 'none',
        'type': str,
        'help': "Zinger removal correction method",
        'choices': ['none', 'standard']},
    'zinger-level-projections': {
        'default': 800.0,
        'type': float,
        'help': 'Expected difference value between outlier value and the median value of the array'},
    'zinger-level-white': {
        'default': 1000.0,
        'type': float,
        'help': 'Expected difference value between outlier value and the median value of the array'},
    'zinger-size': {
        'type': util.positive_int,
        'default': 3,
        'help': "Size of the median filter"},
        }

SECTIONS['flat-correction'] = {
    'flat-correction-method': {
        'default': 'standard',
        'type': str,
        'help': "Flat correction method",
        'choices': ['standard', 'air', 'none']},
    'normalization-cutoff': {
        'default': 1.0,
        'type': float,
        'help': 'Permitted maximum vaue for the normalized data'},
    'air': {
        'type': util.positive_int,
        'default': 10,
        'help': "Number of pixels at each boundary to calculate the scaling factor"},
    'fix-nan-and-inf': {
        'default': False,
        'help': "Fix nan and inf",
        'action': 'store_true'},
    'fix-nan-and-inf-value': {
        'default': 6.0,
        'type': float,
        'help': "Values to be replaced with negative values in array"},
    'minus-log': {
        'default': True,
        'help': "Minus log",
        'action': 'store_true'},
    'sinogram-max-value': {
        'default': float('inf'),
        'help': "Limit the maximum value allowed in the singogram.",
        'type': float},
}

SECTIONS['retrieve-phase'] = {
    'retrieve-phase-method': {
        'default': 'none',
        'type': str,
        'help': "Phase retrieval correction method",
        'choices': ['none', 'paganin']},
    'energy': {
        'default': 20,
        'type': float,
        'help': "X-ray energy [keV]"},
    'propagation-distance': {
        'default': 60,
        'type': float,
        'help': "Sample detector distance [mm]"},
    'pixel-size': {
        'default': 1.17,
        'type': float,
        'help': "Pixel size [microns]"},
    'retrieve-phase-alpha': {
        'default': 0.001,
        'type': float,
        'help': "Regularization parameter"},
    'retrieve-phase-alpha-try': {
        'default': False,
        'help': "When set, multiple reconstruction of the same slice with different alpha coefficient are generated",
        'action': 'store_true'},
    'retrieve-phase-pad': {
        'type': util.positive_int,
        'default': 8,
        'help': "Padding with extra slices in z for phase-retrieval filtering"},
        }

SECTIONS['beam-hardening']= {
    'beam-hardening-method': {
        'default': 'none',
        'type': str,
        'help': "Beam hardening method.",
        'choices':['none','standard']},
    'source-distance': {
        'default': 36.0,
        'type': float,
        'help': 'Distance from source to scintillator in m'},
    'scintillator-material': {
        'default': 'LuAG_Ce',
        'type': str,
        'help': 'Scintillator material for beam hardening',
        'choices': ['LuAG_Ce', 'LYSO_Ce', 'YAG_Ce']},
    'scintillator-thickness': {
        'default': 100.0,
        'type': float,
        'help': 'Scintillator thickness for beam hardening'},
    'center-row': {
        'default': 0.0,
        'type': float,
        'help': 'Row with the center of the vertical fan for beam hardening.'},
    'sample-material': {
        'default': 'Fe',
        'type': str,
        'help': 'Sample material for beam hardening',
        'choices': ['Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-1-auto': {
        'default': False,
        'help': 'If True, read filter 1 from HDF meta data',},
    'filter-1-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 1 material for beam hardening',
        'choices': ['auto','none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-1-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 1 thickness for beam hardening'},
    'filter-2-auto': {
        'default': False,
        'help': 'If True, read filter 2 from HDF meta data',},
    'filter-2-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 2 material for beam hardening',
        'choices': ['auto','none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-2-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 2 thickness for beam hardening'},
    'filter-3-auto': {
        'default': False,
        'help': 'If True, read filter 3 from HDF meta data',},
    'filter-3-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 3 material for beam hardening',
        'choices': ['none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-3-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 3 thickness for beam hardening'},
    }

ALL_PARAMS = ('helical', 'file-reading', 'zinger-removal', 
                'flat-correction', 'retrieve-phase', 'beam-hardening')

NICE_NAMES = ('General', 'Helical', 'File Reading', 'Zinger Removal', 
                'Flat Correction', 'Phase Retrieval', 'Beam Hardening', )

def get_config_name():
    """Get the command line --config option."""
    name = CONFIG_FILE_NAME
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config'):
            if arg == '--config':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--config')[1]
                if name[0] == '=':
                    name = name[1:]
                return name
    return name


def parse_known_args(parser, subparser=False):
    """
    Parse arguments from file and then override by the ones specified on the
    command line. Use *parser* for parsing and is *subparser* is True take into
    account that there is a value on the command line specifying the subparser.
    """
    if len(sys.argv) > 1:
        subparser_value = [sys.argv[1]] if subparser else []
        config_values = config_to_list(config_name=get_config_name())
        values = subparser_value + config_values + sys.argv[1:]
    else:
        values = ""

    return parser.parse_known_args(values)[0]


def config_to_list(config_name=CONFIG_FILE_NAME):
    """
    Read arguments from config file and convert them to a list of keys and
    values as sys.argv does when they are specified on the command line.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()

    if not config.read([config_name]):
        return []

    for section in SECTIONS:
        for name, opts in ((n, o) for n, o in SECTIONS[section].items() if config.has_option(section, n)):
            value = config.get(section, name)

            if value != '' and value != 'None':
                action = opts.get('action', None)

                if action == 'store_true' and value == 'True':
                    # Only the key is on the command line for this action
                    result.append('--{}'.format(name))

                if not action == 'store_true':
                    if opts.get('nargs', None) == '+':
                        result.append('--{}'.format(name))
                        result.extend((v.strip() for v in value.split(',')))
                    else:
                        result.append('--{}={}'.format(name, value))

    return result


def param_from_dxchange(hdf_file, data_path, attr=None, scalar=True, char_array=False):
    """
    Reads a parameter from the HDF file.
    Inputs
    hdf_file: string path or pathlib.Path object for the HDF file.
    data_path: path to the requested data in the HDF file.
    attr: name of the attribute if this is stored as an attribute (default: None)
    scalar: True if the value is a single valued dataset (dafault: True)
    char_array: if True, interpret as a character array.  Useful for EPICS strings (default: False)
    """
    if not os.path.isfile(hdf_file):
        return None
    with h5py.File(hdf_file,'r') as f:
        try:
            if attr:
                return f[data_path].attrs[attr].decode('ASCII')
            elif char_array:
                return ''.join([chr(i) for i in f[data_path][0]]).strip(chr(0))
            elif scalar:
                return f[data_path][0]
            else:
                return None
        except KeyError:
            return None
    

class Params(object):
    def __init__(self, sections=()):
        self.sections = sections + ('general', )

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument('--{}'.format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)

        return parser.parse_args('')


def write(config_file, args=None, sections=None):
    """
    Write *config_file* with values from *args* if they are specified,
    otherwise use the defaults. If *sections* are specified, write values from
    *args* only to those sections, use the defaults on the remaining ones.
    """
    config = configparser.ConfigParser()
    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                value = getattr(args, name.replace('-', '_'))
                if isinstance(value, list):
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''

            prefix = '# ' if value == '' else ''

            if name != 'config':
                config.set(section, prefix + name, str(value))

    with open(config_file, 'w') as f:
        config.write(f)


def log_values(args):
    """Log all values set in the args namespace.

    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('spectrum-filter status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))
        if entries:
            log.info(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                if (value == 'none'):
                    log.warning("  {:<16} {}".format(entry, value))
                elif (value is not False):
                    log.info("  {:<16} {}".format(entry, value))
                elif (value is False):
                    log.warning("  {:<16} {}".format(entry, value))

    log.warning('spectrum-filter status end')
