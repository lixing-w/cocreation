"""Exercise real client processes and the launcher's VLM/materialization barrier."""
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dialogue_generation.parallel_vlm import Case, Settings, load_cases, run_cases


CLIENT = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
def option(name):
    return args[args.index(name)+1]
if '--output-dir' in args:
    directory = Path(option('--output-dir'))
    stage = 'observer' if directory.name == 'observation' else 'dialogue'
else:
    directory = Path(args[0])
    stage = 'preflight' if '--dry-run' in args else 'materialize'
case = directory.parent.name
def event(action):
    record = dict(case=case, stage=stage, action=action, time=time.monotonic(),
                  pid=os.getpid(), args=args,
                  omp=os.environ.get('OMP_NUM_THREADS'), mkl=os.environ.get('MKL_NUM_THREADS'))
    descriptor = os.open(os.environ['TEST_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(descriptor, (json.dumps(record)+'\n').encode())
    os.close(descriptor)
event('start')
durations = json.loads(os.environ.get('TEST_DURATIONS', '{}'))
time.sleep(durations.get(case+':'+stage, 0.08))
if os.environ.get('TEST_FAIL') == case+':'+stage:
    event('failed')
    if stage=='materialize':
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'media_manifest.json').write_text(json.dumps(dict(status='failed',media=[],error='third critique rejected')))
    raise SystemExit(17)
directory.mkdir(parents=True, exist_ok=True)
if stage == 'observer':
    (directory/'observation.json').write_text(json.dumps(dict(assets=[], shots=[])))
elif stage == 'dialogue':
    (directory/'dialogue.json').write_text(json.dumps(dict(turns=[])))
elif stage == 'materialize':
    (directory/'media_manifest.json').write_text(json.dumps(dict(status='complete', media=[])))
else:
    print(json.dumps(dict(dry_run=True)))
event('end')
'''

SERVER = r'''
import json, os, signal, time
from pathlib import Path
def event(action):
    with open(os.environ['TEST_EVENTS'], 'a') as output:
        output.write(json.dumps(dict(case='all', stage='vllm', action=action,
                                     time=time.monotonic(), pid=os.getpid()))+'\n')
def stop(*args):
    event('end')
    Path(os.environ['TEST_READY']).unlink(missing_ok=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(os.environ['TEST_READY']).touch()
event('start')
while True:
    time.sleep(1)
'''


class ParallelVLMTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='parallel-vlm-')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.run = self.repo / 'run'
        self.run.mkdir()
        self.data = self.repo / 'data'
        self.data.mkdir()
        self.events_file = self.repo / 'events.jsonl'
        self.env = dict(os.environ, TEST_EVENTS=str(self.events_file))
        self.env.pop('TEST_DURATIONS', None)
        self.env.pop('TEST_FAIL', None)
        self.cases = [Case(f'case_{i}', 'h3' if i % 2 == 0 else 'openvid',
                           f'case_{i}.json', i, i % 2, 1344, 768)
                      for i in range(3)]
        for name in ('tools/prompt_observer.py', 'scripts/dialogue_generation/generate.py',
                     'scripts/dialogue_generation/materialize.py'):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(CLIENT)
        self.settings = Settings(self.repo, self.run, self.data, 'http://127.0.0.1:8017', concurrency=2)
        self.write_cases()

    def write_cases(self):
        (self.run / 'cases.tsv').write_text(''.join('\t'.join(map(str, vars(c).values()))+'\n'
                                                  for c in self.cases))

    def events(self):
        return [json.loads(line) for line in self.events_file.read_text().splitlines()]

    def run_pool(self, **changes):
        with patch.dict(os.environ, self.env, clear=True):
            asyncio.run(run_cases(self.cases, replace(self.settings, **changes)))

    def assert_reaped(self, events):
        for pid in {event['pid'] for event in events}:
            with self.assertRaises(ProcessLookupError, msg=f'Client {pid} survived cleanup'):
                os.kill(pid, 0)

    def test_bounded_overlap_and_per_case_dependencies(self):
        self.env['TEST_DURATIONS'] = json.dumps({'case_0:observer': 0.8})
        self.run_pool()
        events = self.events()
        active = maximum = 0
        for event in sorted(events, key=lambda e: e['time']):
            active += 1 if event['action'] == 'start' else -1
            maximum = max(maximum, active)
        self.assertEqual((maximum, active), (2, 0))
        for case in self.cases:
            stages = [(e['stage'], e['action']) for e in events if e['case'] == case.id]
            self.assertEqual(stages, [(s, a) for s in ('observer', 'dialogue', 'preflight')
                                      for a in ('start', 'end')])
            self.assertEqual(json.loads((self.run/case.id/'vlm_state.json').read_text())['phase'], 'complete')
            self.assertTrue(json.loads((self.run/case.id/'materialize-plan.json').read_text())['dry_run'])
        def timestamp(case, stage, action):
            return next(e['time'] for e in events if (e['case'], e['stage'], e['action']) == (case, stage, action))
        self.assertLess(timestamp('case_1', 'dialogue', 'start'), timestamp('case_0', 'observer', 'end'))
        for event in events:
            self.assertEqual((event['omp'], event['mkl']), ('1', '1'))
            if event['stage'] != 'preflight':
                args = event['args']
                self.assertEqual(args[args.index('--base-url')+1], self.settings.base_url)
        summary = json.loads((self.run/'vlm_summary.json').read_text())
        self.assertEqual((summary['status'], summary['effective_concurrency']), ('complete', 2))

    def test_resume_and_base_forty_policy_are_preserved(self):
        case_dir = self.run/'case_0'
        for folder, filename in (('observation', 'observation.json'), ('dialogue', 'dialogue.json')):
            (case_dir/folder).mkdir(parents=True)
            (case_dir/folder/filename).write_text('{}')
        self.run_pool(concurrency=1, match_video_brightness=False)
        events = self.events()
        self.assertFalse(any(e['case'] == 'case_0' and e['stage'] == 'observer' for e in events))
        dialogue = next(e for e in events if e['case'] == 'case_0' and e['stage'] == 'dialogue')
        self.assertIn('--resume', dialogue['args'])
        for event in events:
            if event['stage'] == 'preflight':
                args = event['args']
                self.assertEqual(args[args.index('--video-steps')+1], '40')
                self.assertNotIn('--new-video-turbo-lora', args)
                self.assertNotIn('--video-turbo-lora', args)
                self.assertIn('--no-match-video-brightness', args)
        active = 0
        for event in sorted(events, key=lambda e: e['time']):
            active += 1 if event['action'] == 'start' else -1
            self.assertLessEqual(active, 1)

    def test_failure_cancels_other_clients_and_does_not_start_queued_case(self):
        self.env.update(TEST_FAIL='case_0:dialogue', TEST_DURATIONS=json.dumps({'case_1:observer': 10}))
        with self.assertRaisesRegex(RuntimeError, 'case_0.*dialogue_and_generation_prompts'):
            self.run_pool()
        events = self.events()
        self.assert_reaped(events)
        self.assertFalse(any(e['case'] == 'case_2' or e['stage'] == 'preflight' for e in events))
        self.assertEqual(json.loads((self.run/'case_1/vlm_state.json').read_text())['phase'], 'cancelled')
        self.assertEqual(json.loads((self.run/'vlm_summary.json').read_text())['status'], 'failed')

    def test_sigterm_reaps_clients(self):
        self.env['TEST_DURATIONS'] = json.dumps({f'case_{i}:observer': 10 for i in range(3)})
        command = [sys.executable, str(ROOT/'scripts/dialogue_generation/parallel_vlm.py'),
                   '--repo', str(self.repo), '--run-dir', str(self.run), '--data-dir', str(self.data),
                   '--base-url', self.settings.base_url, '--concurrency', '2']
        process = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic()+10
            while not self.events_file.exists() or len(self.events()) < 2:
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail('VLM clients failed to start')
                time.sleep(0.02)
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143, output)
            self.assert_reaped(self.events())
            self.assertEqual(json.loads((self.run/'vlm_summary.json').read_text())['status'], 'cancelled')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_invalid_cases_or_concurrency_launch_nothing(self):
        cases_file = self.run/'cases.tsv'
        self.assertEqual(load_cases(cases_file), self.cases)
        lines = [line.split('\t') for line in cases_file.read_text().splitlines()]
        cases_file.write_text(''.join('\t'.join([*row[:5], 'portrait', *row[5:]])+'\n'
                                     for row in lines))
        self.assertEqual(load_cases(cases_file), self.cases)
        self.cases.append(self.cases[0])
        self.write_cases()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            load_cases(self.run/'cases.tsv')
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.run_pool(concurrency=0)
        self.assertFalse(self.events_file.exists())

    def prepare_launcher(self):
        # Exercise the actual shell launcher with local fake CLIs, never a GPU/model.
        launcher = self.repo/'run_actual_prompt_pipeline.sh'
        shutil.copyfile(ROOT/launcher.name, launcher)
        shutil.copyfile(ROOT/'scripts/dialogue_generation/parallel_vlm.py',
                        self.repo/'scripts/dialogue_generation/parallel_vlm.py')
        package = self.repo/'cocreation'
        package.mkdir()
        (package/'__init__.py').write_text('')
        shutil.copyfile(ROOT/'cocreation/real_prompt_policy.py', package/'real_prompt_policy.py')
        for name in ('scripts/dialogue_generation/planner_rules.py','tools/media_review.py','tools/prompt_reasoner.py','tools/placeholder_prompter.py','tools/dialogue_realizer.py','tools/dialogue_verifier.py','scripts/dialogue_generation/rewrite_prompts.py','scripts/dialogue_generation/formal_sequence.py','tools/prompt_observer_prompts.py', 'scripts/dialogue_generation/asset_variations.py',
                     'cocreation/video_tone.py', 'cocreation/h3_policy.py', 'cocreation/h3.py',
                     'cocreation/models.py', 'cocreation/comfy_worker.py'):
            (self.repo/name).touch()
        # The launcher's provenance step reads these files before starting the clients.
        for name in ('tools/turn_media_review.py', 'scripts/dialogue_generation/reviewed_turn.py',
                     'scripts/dialogue_generation/generation.py', 'scripts/dialogue_generation/catalog.py',
                     'tools/video_editing.py', 'tools/prompt_fixer.py', 'cocreation/qwen_client.py'):
            (self.repo/name).touch()
        shutil.copyfile(ROOT/'cocreation/qwen_inputs.py', package/'qwen_inputs.py')
        server = self.repo/'scripts/qwen/serve.py'
        server.parent.mkdir()
        server.write_text(SERVER)
        runtime = self.repo/'scratch/runtime/qwen3.6-35b-a3b'
        (runtime/'vllm/bin').mkdir(parents=True)
        (runtime/'vllm/bin/python').symlink_to(sys.executable)
        (runtime/'server.json').write_text('{}')
        manifest = []
        for c in self.cases:
            raw = b'{}'
            (self.data/c.input_file).write_bytes(raw)
            manifest.append(dict(id=c.id, source_format=c.source_format, input_file=c.input_file,
                                 input_sha256=hashlib.sha256(raw).hexdigest(), seed=c.seed, reworks=c.reworks,
                                 video_width=c.width, video_height=c.height))
        manifest[0]['aspect_ratio_rule'] = 'portrait'
        (self.data/'manifest.json').write_text(json.dumps(dict(cases=manifest)))
        profile = self.repo/'etc/profile.d/conda.sh'
        profile.parent.mkdir(parents=True)
        profile.write_text('conda() { :; }\n')
        bash_env = self.repo/'bash-env'
        bash_env.write_text('module() { :; }\n')
        bin_dir = self.repo/'bin'
        bin_dir.mkdir()
        (bin_dir/'python').symlink_to(sys.executable)
        scripts = {'nvidia-smi': '#!/bin/sh\necho NVIDIA B200\n',
                   'curl': '#!/bin/sh\ntest -f "$TEST_READY"\n',
                   'sleep': f'#!{sys.executable}\nimport time; time.sleep(0.02)\n'}
        for name, content in scripts.items():
            (bin_dir/name).write_text(content)
            (bin_dir/name).chmod(0o755)
        self.env.update(COCREATION_REPO=str(self.repo), DATA_DIR=str(self.data), RUN_DIR=str(self.run),
                        EXPECTED_CASES=str(len(self.cases)), VLM_CONCURRENCY='2', H3_VIDEO_STEPS='40',
                        MAMBA_ROOT_PREFIX=str(self.repo), BASH_ENV=str(bash_env),
                        TEST_READY=str(self.repo/'ready'), PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
        return launcher

    def run_launcher(self):
        launcher = self.prepare_launcher()
        return subprocess.run(['bash', str(launcher)], env=self.env, cwd=self.repo,
                              capture_output=True, text=True, timeout=30)

    def test_launcher_waits_for_all_preflights_then_materializes_serially(self):
        self.env['TEST_DURATIONS'] = json.dumps({'case_2:preflight': 0.5})
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        events = self.events()
        preflight_end = max(e['time'] for e in events if e['stage'] == 'preflight' and e['action'] == 'end')
        server_end = next(e['time'] for e in events if e['stage'] == 'vllm' and e['action'] == 'end')
        media = sorted([e for e in events if e['stage'] == 'materialize'], key=lambda e: e['time'])
        self.assertLess(preflight_end, server_end)
        self.assertLess(media[-1]['time'],server_end)
        self.assertEqual([(e['case'], e['action']) for e in media],
                         [(c.id, a) for c in self.cases for a in ('start', 'end')])
        for event in events:
            if event['stage'] == 'dialogue':
                self.assertNotIn('--aspect-ratio-rule', event['args'])
            if event['stage'] in ('preflight', 'materialize'):
                self.assertEqual(event['args'][event['args'].index('--video-width')+1], '1344')
                self.assertEqual(event['args'][event['args'].index('--video-height')+1], '768')
                args = event['args']
                self.assertEqual(args[args.index('--video-steps')+1], '40')
                self.assertNotIn('--new-video-turbo-lora', args)
                self.assertNotIn('--video-turbo-lora', args)
        material_args=next(e['args'] for e in events if e['stage']=='materialize')
        self.assertIn('--shared-vlm',material_args)
        self.assertEqual(material_args[material_args.index('--review-base-url')+1],'http://127.0.0.1:8017')
        execution=json.loads((self.run/'vlm_execution.json').read_text())
        self.assertTrue(execution['persistent_through_materialization'])
        self.assertEqual(execution['kv_cache_memory_gib'],16)
        policy = json.loads((self.run/'inference_policy.json').read_text())
        self.assertEqual(policy['new_video'], dict(turbo=False, steps=40))
        self.assertEqual(policy['video_edit']['steps'], 40)
        self.assertFalse(policy['video_edit']['turbo'])
        self.assert_reaped(events)
        self.assertEqual(json.loads((self.run/'pipeline_state.json').read_text())['phase'], 'complete')

    def test_media_rejection_terminates_only_that_sequence(self):
        self.env['TEST_FAIL']='case_0:materialize'
        result=self.run_launcher()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        events=self.events()
        self.assertTrue(any(e['case']=='case_1' and e['stage']=='materialize' and e['action']=='end' for e in events))
        results=json.loads((self.run/'results.json').read_text())
        self.assertEqual(results[0]['materialization'],'failed')
        self.assertEqual(results[1]['materialization'],'complete')
        self.assertEqual(json.loads((self.run/'pipeline_state.json').read_text())['phase'],'complete_with_failures')

    def test_launcher_failure_never_materializes_and_stops_server(self):
        self.env.update(TEST_FAIL='case_0:dialogue', TEST_DURATIONS=json.dumps({'case_1:observer': 10}))
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        events = self.events()
        self.assertFalse(any(e['stage'] == 'materialize' for e in events))
        self.assertTrue(any(e['stage'] == 'vllm' and e['action'] == 'end' for e in events))
        self.assert_reaped(events)
        self.assertFalse((self.run/'text_phase_complete.txt').exists())
        self.assertEqual(json.loads((self.run/'pipeline_state.json').read_text())['phase'], 'failed')


if __name__ == '__main__':
    unittest.main()
