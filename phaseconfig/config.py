import sys
import argparse
import logging
import ConfigParser as configparse
from collections import OrderedDict

from phaseconfig.util import positive_int

LOG = logging.getLogger(__name__)
NAME = "phasefilter.conf"
SECTIONS = OrderedDict()

SECTIONS['general'] = {
    'config': {
        'default': NAME,
        'type': str,
        'help': "File name of configuration",
        'metavar': 'FILE'},
    'output': {
        'default': 'profiles.png',
        'type': str,
        'help': "A format-specified file path to save the comparative plot",
        'metavar': 'PATH'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'log': {
        'default': None,
        'type': str,
        'help': "Filename of optional log",
        'metavar': 'FILE'}
}

SECTIONS['reading'] = {
    'y': {
        'type': positive_int,
        'default': 0,
        'help': 'Vertical coordinate from where to start reading the input image'},
    'height': {
        'default': None,
        'type': positive_int,
        'help': "Number of rows which will be read"},
    'bitdepth': {
        'default': 32,
        'type': positive_int,
        'help': "Bit depth of raw files"},
    'y-step': {
        'type': positive_int,
        'default': 1,
        'help': "Read every \"step\" row from the input"},
    'start': {
        'type': positive_int,
        'default': 0,
        'help': 'Offset to the first read file'},
    'number': {
        'type': positive_int,
        'default': None,
        'help': 'Number of files to read'},
    'step': {
        'type': positive_int,
        'default': 1,
        'help': 'Read every \"step\" file'},
    'resize': {
        'type': positive_int,
        'default': None,
        'help': 'Bin pixels before processing'},
    'retries': {
        'type': positive_int,
        'default': 0,
        'metavar': 'NUMBER',
        'help': 'How many times to wait for new files'},
    'retry-timeout': {
        'type': positive_int,
        'default': 0,
        'metavar': 'TIME',
        'help': 'How long to wait for new files per trial'}
}

SECTIONS['power-spectral-density'] = {
    'projections': {
        'default: None',
        'type': str,
        'help': "Location with corrected projections"},
    'power-spectral-density': {
        'default': 'psd.tif',
        'type': str,
        'help': "Path to location or format-specified file path "
                "for storing a power spectral density image.",
        'metavar': 'PATH'}
}

SECTIONS['retrieve-phase'] = {
    'retrieval-method': {
        'coices': ['tie', 'ctf', 'ctfhalfsin', 'qp', 'qphalfsine', 'qp2'],
        'default': 'qp',
        'help': 'Phase retrieval method.'},
    'energy' : {
        'default': None,
        'type': float,
        'help': "X-ray energy [keV]"},
    'propagation-distance': {
        'default': None,
        'type': float,
        'help': "Sample <-> detector distance [m]"},
    'pixel-size': {
        'default': 1e-6,
        'type': float,
        'help': "Pixel size [m]"},
    'regularization-rate': {
        'default': 2,
        'type': float,
        'help': "Regularization rate (typical values between [2,3])"},
    'retrieval-padded-width': {
        'default': 0,
        'type': positive_int,
        'help': "Padded width used for phase retrieval"},
    'retrieval-padded-height': {
        'default': 0,
        'type': positive_int,
        'help': "Padded height used for phase retrieval"},
    'retrieval-padding-mode': {
        'coices': ['none', 'clamp', 'clamp_to_edge', 'repeat'],
        'default': 'clamp_to_edge',
        'help': "Padded values assignment"},
    'thresholding-rate': {
        'default': 0.001,
        'type': float,
        'help': "thresholding rate (typical values betwen [0.01, 0.1])"}
}

NICE_NAMES = ('General', 'Power Spectrum Density', 'Analysis of phase retrieval')

def get_config_name():
    """Get the command line --config option."""
    name = NAME
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


def config_to_list(config_name=NAME):
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

            if value is not '' and value != 'None':
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


class Params(object):
    def __init__(self, sections=()):
        self.sections = sections + ('general', 'reading')

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

            prefix = '# ' if value is '' else ''

            if name != 'config':
                config.set(section, prefix + name, value)

    with open(config_file, 'wb') as f:
        config.write(f)


def log_values(args):
    """Log all values set in the args namespace.

    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))

        if entries:
            LOG.debug(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                LOG.debug("  {:<16} {}".format(entry, value))
