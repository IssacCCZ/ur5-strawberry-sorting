"""Installer control-flow checks: no sudo, apt changes, or external downloads."""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'install_ubuntu2404.sh'


class InstallerTests(unittest.TestCase):
    def shell(self, body, directory):
        return subprocess.run(['bash', '-c', f'source {shlex.quote(str(SCRIPT))}\n{body}'],
                              cwd=directory, text=True, capture_output=True, check=False,
                              env={**os.environ, 'XDG_CACHE_HOME': str(directory / 'cache')})

    def test_plan_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.shell('main --plan --stage all --install-driver --vision cuda', root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('5.1.0', result.stdout)
            self.assertIn('hw2_vision_cu128', result.stdout)
            self.assertEqual(list(root.iterdir()), [])

    def test_bad_arguments_fail_without_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for args in ('--stage bogus', '--vision', '--test invalid', '--install-root relative --plan'):
                result = self.shell(f'main {args}', root)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_other_loaded_driver_never_reaches_sudo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.shell('''
loaded_driver() { printf '610.43.02\n'; }
sudo() { touch SHOULD_NOT_RUN; }
install_driver=true
system_install
''', root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('不会替换', result.stderr)
            self.assertFalse((root / 'SHOULD_NOT_RUN').exists())

    def test_missing_driver_requires_explicit_install_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.shell('''
loaded_driver() { :; }
sudo() { touch SHOULD_NOT_RUN; }
system_install
''', root)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'SHOULD_NOT_RUN').exists())

    def test_download_resumes_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.bin'
            content = bytes(range(256)) * 100
            source.write_bytes(content)
            destination = root / 'cache' / 'payload.bin'
            destination.parent.mkdir()
            destination.with_suffix('.bin.part').write_bytes(content[:1234])
            result = self.shell(f'fetch {shlex.quote(source.as_uri())} {shlex.quote(str(destination))}', root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(destination.with_suffix('.bin.part').exists())

    def test_incomplete_isaac_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / 'isaacsim-5.1.0'
            target.mkdir()
            (target / 'keep.txt').write_text('user file')
            result = self.shell(f'isaac_dir={shlex.quote(str(target))}\ninstall_isaac', root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((target / 'keep.txt').read_text(), 'user file')

    def test_launcher_quotes_paths_and_selects_installed_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Literal shell syntax in a path must never become executable code.
            install = root / 'install $(touch INJECTED)'
            log = root / 'logs with spaces'
            log.mkdir()
            result = self.shell(f'''
parse_args --install-root {shlex.quote(str(install))} --vision cuda
log_dir={shlex.quote(str(log))}
write_launcher
source "$log_dir/environment.sh"
[[ "$ISAAC_SIM_PATH" == "$isaac_dir" ]]
[[ "$HW2_VISION_PYTHON" == "$vision_prefix/bin/python" ]]
bash -n "$log_dir/launch.sh"
''', root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'INJECTED').exists())
            self.assertIn('--device 0', (log / 'launch.sh').read_text())


if __name__ == '__main__':
    unittest.main()
