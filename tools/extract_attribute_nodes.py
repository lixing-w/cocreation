"""Compatibility entry point: now returns prose, never nested attribute JSON."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Reference
from tools.extractor import extractor

extract_attribute_nodes = extractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--kind', choices=('image', 'video'), required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-system-cache', dest='use_system_cache', action='store_false')
    args = vars(parser.parse_args())
    ref = Reference(args.pop('artifact_id'), args.pop('path'), args.pop('kind'))
    print(extractor(ref, **args) or 'NO_OBSERVATION')
