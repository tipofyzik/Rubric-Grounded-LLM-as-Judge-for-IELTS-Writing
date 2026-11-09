"""Build a matching AES notebook and verified prepared-data snapshot, without training."""
from typing import Any
import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4
import zipfile

ROOT=next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path/'code/shared/paths.yaml').is_file())
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/shared')]


def build(output: Any) -> Any:
    """Build.

    Args:
        output (Any): Output used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from ielts_judge.data.Preparation import load_prepared, file_hash
    from ielts_judge.Paths import PATHS
    release=Path(PATHS['datasets']['prepared'])
    _,_,manifest=load_prepared(release)
    files={}
    for folder in ('code/shared/ielts_judge','code/offline_research'):
        for path in (ROOT/folder).rglob('*.py'):
            if set(path.relative_to(ROOT).parts).intersection({'tools','notebooks','legacy','__pycache__','.cache','artifacts'}):continue
            files[path.relative_to(ROOT).as_posix()]=path
    for name in ('requirements.txt','code/shared/paths.yaml','code/shared/config.yaml','code/offline_research/config.yaml','code/shared/resources/ielts_criteria.yaml','code/shared/resources/model_registry.yaml'):
        files[name]=ROOT/name
    for name in ('manifest.json',*manifest['artifacts']):
        path=release/name;files[path.relative_to(ROOT).as_posix()]=path
    hashes={name:file_hash(path) for name,path in sorted(files.items())}
    destination=Path(output)/uuid4().hex[:12];destination.mkdir(parents=True,exist_ok=False)
    archive=destination/'aes_training_bundle.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for name,path in sorted(files.items()):target.write(path,name)
    document=json.loads((ROOT/'code/offline_research/notebooks/AESTraining.ipynb').read_text(encoding='utf-8'))
    sha=file_hash(archive)
    for cell in document['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        if 'project_root = prepare_project(' in source:
            tail=source[source.index('project_root = prepare_project('):]
            cell['source']='bundle_path = None\nexpected_sha256 = '+repr(sha)+'\nexpected_files = '+repr(hashes)+'\n'+tail
        cell.update(outputs=[],execution_count=None)
    (destination/'AESTraining.ipynb').write_text(json.dumps(document,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (destination/'AESBundleManifest.json').write_text(json.dumps({'archive_sha256':sha,'files':hashes},indent=2)+'\n',encoding='utf-8')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results/aes/kaggle')
    print(build(parser.parse_args().output))
