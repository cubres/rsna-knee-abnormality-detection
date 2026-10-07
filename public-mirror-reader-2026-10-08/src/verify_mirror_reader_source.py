"""Small source/configuration checks; no checkpoint, native tensor or network execution."""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
path = ROOT / 'public_mirror_reader.py'
tree = ast.parse(path.read_text())
compile(tree, str(path), 'exec')
spec = importlib.util.spec_from_file_location('owned_knee_mirror_source_fixture', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
def check(name, value):
    if not value:
        raise RuntimeError(name)
    checks.append(name)

check('exact96slice_slot_budget', module.SLOTS == (('Sagittal', 1, 26), ('Sagittal', 0, 22),
      ('Coronal', 1, 18), ('Coronal', 0, 12), ('Axial', -1, 18)) and sum(x[2] for x in module.SLOTS) == 96)
check('all94adjacent_windows', module.WINDOW_COUNT == 94)
check('384to320_center32', module.CORPUS_RES == 384 and module.INPUT_RES == 320)
check('140mm_physicalcrop', module.CROP_MM == 140.0)
check('source_and_asset_version_bind', module.PUBLIC_VERSION == 2 and module.PUBLIC_SESSION_ID == 356175397
      and module.DATASET_VERSION == 1 and module.CHECKPOINT_SHA256 == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef')
for center, expected in ((0,'channels'),(25,'channels'),(26,'channels'),(47,'channels'),
                         (48,'width'),(65,'width'),(66,'width'),(77,'width'),(78,'width'),(95,'width')):
    check('actual_mirror_helper_center_' + str(center), module.mirror_axis_for_center(center) == expected)
for index, bad in enumerate((-1, 96, None, '48', True)):
    try:
        module.mirror_axis_for_center(bad)
    except RuntimeError:
        check('invalid_center_rejected_' + str(index), True)
    else:
        raise RuntimeError('Invalid centre accepted')
check('mirror_runtime_calls_checked_helper', any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
      and n.func.id == 'mirror_axis_for_center' for n in ast.walk(tree)))
check('no_native_dependency_loaded_on_import', all(name not in sys.modules for name in ('torch','timm','pydicom')))
check('no_fork_or_subprocess_imports', not any(isinstance(n, ast.Import) and any(a.name in ('subprocess','multiprocessing')
      for a in n.names) for n in ast.walk(tree)))
check('gates_survive_optimization', not any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
record = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status':'PASS_SOURCE_CONFIGURATION',
          'optimized':bool(sys.flags.optimize), 'checks':checks, 'check_count':len(checks),
          'module_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'checkpoint_downloads':0, 'model_execution':False, 'native_equivalence_proven':False,
          'remote_mutations':0}
with args.receipt.open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(record, sort_keys=True))
