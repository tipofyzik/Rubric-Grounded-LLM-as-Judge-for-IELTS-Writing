"""User-facing installation and component boundaries of the source distribution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path/'Manage.py').is_file())
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/shared')]
spec=importlib.util.spec_from_file_location('project_manager',ROOT/'Manage.py')
manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)


class DistributionTests(unittest.TestCase):
    """Manage distribution tests through the operations defined by this component."""
    def test_three_components_have_no_original_path_facades(self) -> None:
        """Verify that three components have no original path facades.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual({candidate_path.name for candidate_path in (ROOT/'code').iterdir() if candidate_path.is_dir()}, {'public_demo','local_backup','offline_research','shared'})
        self.assertFalse((ROOT/'code/App.py').exists())
        self.assertFalse((ROOT/'code/ielts_judge').exists())
        self.assertTrue((ROOT/'code/offline_research/notebooks/AESTraining.ipynb').is_file())

    def test_settings_precedence_and_import_secret_are_explicit(self) -> None:
        """Verify that settings precedence and import secret are explicit.

        Returns:
            None: This operation returns no value.
        """
        from ielts_judge.Access import read_access
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root=Path(temporary)
            (root/'.env').write_text('HF_TOKEN=file-token\nIELTS_MODEL_KEY=chosen\nSUPABASE_SECRET_KEY=import-only\n',encoding='utf-8')
            values=read_access('base',root=root)
            self.assertEqual(values['IELTS_MODEL_KEY'],'chosen')
            self.assertNotIn('SUPABASE_SECRET_KEY',values)
            os.environ['HF_TOKEN']='process-token'
            self.assertEqual(read_access('base',root=root)['HF_TOKEN'],'process-token')

    def test_space_has_no_author_fallback(self) -> None:
        """Verify that space has no author fallback.

        Returns:
            None: This operation returns no value.
        """
        from local_backup.Runtime import check_space, LauncherError
        with self.assertRaises(LauncherError) as error:
            check_space({'HF_TOKEN':'never-print'})
        self.assertEqual(error.exception.code,'invalid_space')
        self.assertNotIn('never-print',str(error.exception))

    def test_selected_model_reaches_real_remote_adapter(self) -> None:
        """Verify that selected model reaches real remote adapter.

        Returns:
            None: This operation returns no value.
        """
        import yaml
        from ielts_judge.ModelRegistry import ModelRegistry
        from local_backup import Runtime as runtime
        configuration=yaml.safe_load((ROOT/'code/shared/resources/model_registry.yaml').read_text(encoding='utf-8'))
        configuration['models']['chosen']=dict(configuration['models']['qwen25_1p5b'],repo_id='owner/model',label='Chosen')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'models.yaml';path.write_text(yaml.safe_dump(configuration),encoding='utf-8')
            registry=ModelRegistry(path)
            with patch.object(runtime,'ModelRegistry',return_value=registry), patch('transformers.AutoTokenizer.from_pretrained') as load:
                judge=runtime.RemoteJudge({'HF_TOKEN':'never-print','IELTS_MODEL_KEY':'chosen'},'https://example.invalid')
            self.assertEqual(judge.profile.key,'chosen')
            self.assertEqual(load.call_args.args[0],'owner/model')
            self.assertTrue(load.call_args.kwargs['local_files_only'])

    def test_source_export_excludes_private_inputs_and_does_not_overwrite(self) -> None:
        """Verify that source export excludes private inputs and does not overwrite.

        Returns:
            None: This operation returns no value.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name, content in {'Manage.py':'pass','.env.example':'HF_TOKEN=','code/local_backup/App.py':'pass',
                                  '.env':'HF_TOKEN=PRIVATE_VALUE','data/raw.csv':'PRIVATE_DATA',
                                  'results/model.pt':'PRIVATE_WEIGHT','code/offline_research/notebooks/artifacts/run.py':'PRIVATE_OUTPUT'}.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
            with patch.object(manager,'ROOT',root):
                manager.source_archive(root/'source.zip')
                with self.assertRaises(FileExistsError):manager.source_archive(root/'source.zip')
            with zipfile.ZipFile(root/'source.zip') as archive:
                self.assertIn('code/local_backup/App.py',archive.namelist())
                self.assertIn('.env.example',archive.namelist())
                self.assertFalse(any('PRIVATE_' in archive.read(name).decode('utf-8') for name in archive.namelist()))

    def test_manager_help_and_install_plan_work_outside_checkout(self) -> None:
        """Verify that manager help and install plan work outside checkout.

        Returns:
            None: This operation returns no value.
        """
        with tempfile.TemporaryDirectory() as temporary:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'Manage.py'),'install','local','--dry-run'],cwd=temporary,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(Path(plan['venv']),ROOT/'.venv')
        self.assertIn('code/local_backup/requirements.txt',plan['commands'][-1])


if __name__=='__main__':unittest.main(verbosity=2)
