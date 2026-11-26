"""Real ffmpeg materialization with injected generators; no model loading."""
from copy import deepcopy
import array
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.materialize import materialize, file_hash, duration
from tools.video_editing import concatenate, probe


def write_json(path, data):
    Path(path).write_text(json.dumps(data))


def movie(path, color, frequency, seconds=1, size='96x64'):
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
        f'color={color}:s={size}:r=10:d={seconds}','-f','lavfi','-i',
        f'sine=frequency={frequency}:duration={seconds}','-c:v','libx264',
        '-pix_fmt','yuv420p','-c:a','aac','-shortest',str(path)],check=True)


class FakeModels:
    def __init__(self, fail_prompt=None):
        self.calls = []
        self.closed = []
        self.fail_prompt = fail_prompt

    def __call__(self, backend, model_id, output_dir):
        owner = self
        class Model:
            def __init__(self):
                self.output_dir=Path(output_dir)
            def __call__(self, prompt, references, **options):
                owner.calls.append(dict(backend=backend, model_id=model_id, prompt=prompt,
                    references=references, options=options, reference_seconds=[duration(r.path) if r.kind=='video' else None for r in references]))
                if prompt == owner.fail_prompt:
                    raise RuntimeError('injected generator interruption')
                directory = Path(self.output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                if backend == 'flux':
                    from PIL import Image
                    path = directory/f'fake-{len(owner.calls)}.png'
                    Image.new('RGB',(64,64),'red' if prompt == 'variant' else 'blue').save(path)
                else:
                    path = directory/f'fake-{len(owner.calls)}.mp4'
                    movie(path,'green',330,seconds=options['seconds'],size='128x64')
                return dict(status='ok',path=str(path))
            def close(self):
                owner.closed.append(backend)
        return Model()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'dialogue'
        self.run.mkdir()
        self.a, self.b = self.root/'a.mp4', self.root/'b.mp4'
        movie(self.a,'red',440)
        movie(self.b,'blue',880)
        self.target = concatenate([self.a,self.b],self.root/'target.mp4')
        self.original = '# Dialogue\n\nAssistant: [[image_0001]]\n\nReference: [[video_0002]]\n\nResult: [[video_0003]]\n\nTarget: [[video_0009]]\n'
        (self.run/'dialogue.md').write_text(self.original)
        write_json(self.run/'dialogue.json',dict(media_generated=False,turns=[]))
        write_json(self.run/'observation.json',dict(shots=[
            dict(id='S1',start=0,end=1,description='red'),
            dict(id='S2',start=1,end=2,description='blue')]))
        self.catalog = [
            dict(placeholder='[[image_0001]]',kind='image',operation='local_edit',backend='flux',
                model_id='black-forest-labs/FLUX.2-dev',prompt='variant',
                reference_materials=['[[image_0002]]'],
                references=[dict(placeholder='[[image_0002]]',kind='image',label='<Image 1>')]),
            dict(placeholder='[[video_0003]]',kind='video',operation='concatenate',prompt=None,
                reference_materials=['[[video_0001]]','[[video_0002]]'],
                clips=[dict(placeholder='[[video_0002]]'),dict(placeholder='[[video_0001]]')]),
            dict(placeholder='[[image_0002]]',kind='image',operation='asset_reconstruction',backend='flux',
                model_id='black-forest-labs/FLUX.2-dev',prompt='reconstruct <Image 1>',
                reference_materials=['[[video_0009]]'],
                references=[dict(placeholder='[[video_0009]]',kind='image',frame_seconds=.5,label='<Image 1>')]),
            dict(placeholder='[[image_0003]]',kind='image',operation='local_edit',backend='flux',
                prompt='edit intermediate frame <Image 1>',reference_materials=['[[video_0002]]'],
                references=[dict(placeholder='[[video_0002]]',kind='image',frame_seconds=0.,label='<Image 1>')]),
            dict(placeholder='[[video_0001]]',kind='video',operation='extract_shot',shot_id='S1',prompt=None,
                reference_materials=['[[video_0009]]']),
            dict(placeholder='[[video_0002]]',kind='video',operation='video_edit',backend='vdn_h3',
                prompt='edit <Video 1> with <Picture 1>',seconds=5.,output_seconds=1.,shot_id='S2',
                reference_materials=['[[image_0001]]'],dependencies=['[[video_0009]]','[[image_0001]]'],
                references=[dict(placeholder='[[video_0009]]',kind='video',shot_id='S2',label='<Video 1>'),
                            dict(placeholder='[[image_0001]]',kind='image',label='<Picture 1>')]),
            dict(placeholder='[[video_0004]]',kind='video',operation='copy',prompt=None,
                reference_materials=['[[video_0003]]']),
            dict(placeholder='[[image_0004]]',kind='image',operation='extract_frame',prompt=None,frame_seconds=.2,
                reference_materials=['[[video_0002]]']),
            dict(placeholder='[[video_0009]]',kind='video',operation='source',prompt=None,
                source_video=str(self.target),reference_materials=[])]
        self.persist()

    def persist(self):
        write_json(self.run/'placeholders.json',self.catalog)

    def test_assembly_receipt_matches_actual_python_call_without_extra_generation(self):
        opening="This turn you only asked me to join existing clips, so I'm not generating new media and the rule requiring a winter background for new media does not apply."
        write_json(self.run/'dialogue.json',dict(turns=[dict(id=1,role='user',content='Join the clips.'),
            dict(id=2,role='assistant',content='Here is the assembly.',generation_opening=opening,
                 output_media=['[[video_0003]]'])]))
        assembly=next(row for row in self.catalog if row['placeholder']=='[[video_0003]]')
        assembly['operation_reasoner']=dict(think='I will join the second clip and then the first clip.')
        self.persist()
        models=FakeModels()
        with patch('tools.video_editing.concatenate',wraps=concatenate) as actual:
            result=self.execute(models)
        record=result['media']['[[video_0003]]']
        self.assertEqual(actual.call_count,1)
        self.assertEqual(record['tool_call']['arguments'],actual.call_args.kwargs)
        self.assertEqual(len(models.calls),sum(row.get('backend') is not None for row in self.catalog))
        self.assertFalse(record.get('attempts'))
        self.assertFalse(record.get('actual_prompt'))
        exported=json.loads((self.run/'sequence.json').read_text())['turns'][1]['segments']
        self.assertEqual(exported[0]['text'],opening)
        self.assertEqual(exported[1]['tool_calls'],[record['tool_call']])

    def execute(self, models=None, **options):
        return materialize(self.run,model_factory=models or FakeModels(),progress=False,**options)

    def test_complete_dependency_graph_and_html_with_real_media(self):
        models = FakeModels()
        result = self.execute(models)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['completed'],len(self.catalog))
        self.assertEqual(result['construction_order'][0],'[[video_0009]]')
        self.assertEqual(file_hash(self.target),file_hash(self.run/'media/video_0009.mp4'))
        self.assertEqual((self.run/'dialogue.placeholders.md').read_text(),self.original)
        rendered = (self.run/'dialogue.md').read_text()
        self.assertIn('<img src="media/image_0001.png"',rendered)
        self.assertIn('<video src="media/video_0002.mp4" controls',rendered)
        self.assertNotIn('[[image_',rendered)
        self.assertNotIn('[[video_',rendered)
        self.assertTrue(json.loads((self.run/'dialogue.json').read_text())['media_generated'])
        self.assertEqual(len(models.calls),4)
        self.assertEqual(models.calls[0]['model_id'],'black-forest-labs/FLUX.2-dev')
        video = next(c for c in models.calls if c['backend']=='vdn_h3')
        self.assertEqual([r.kind for r in video['references']],['video','image'])
        self.assertEqual(video['prompt'],'edit <Video 1> with <Picture 1>')
        self.assertAlmostEqual(video['reference_seconds'][0],1,places=1)
        image_from_video = models.calls[-1]
        self.assertEqual(image_from_video['references'][0].kind,'image')
        self.assertEqual(Path(image_from_video['references'][0].path).suffix,'.png')
        self.assertEqual(models.closed,['flux','vdn_h3','flux'])
        self.assertAlmostEqual(duration(self.run/'media/video_0002.mp4'),1,places=1)
        self.assertAlmostEqual(duration(self.run/'media/video_0003.mp4'),2,places=1)
        output=probe(self.run/'media/video_0002.mp4')
        v=next(s for s in output['streams'] if s['codec_type']=='video')
        self.assertEqual((v['width'],v['height']),(96,64))
        # Deterministic concatenation obeys clips order, not reference_materials.
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(self.run/'media/video_0003.mp4'),
            '-vf',"select='eq(n,5)+eq(n,15)',scale=1:1",'-fps_mode','vfr','-f','rawvideo','-pix_fmt','rgb24','-'])
        self.assertEqual([max(range(3),key=lambda c:raw[i+c]) for i in range(0,len(raw),3)],[1,0])
        # Visual edits retain the source-shot audio (880 Hz), not fake-model audio (330 Hz).
        audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(self.run/'media/video_0002.mp4'),
            '-f','s16le','-ac','1','-ar','8000','-'])
        samples=array.array('h',audio)[1600:6400]
        hz=sum(a<=0<b for a,b in zip(samples,samples[1:]))/.6
        self.assertAlmostEqual(hz,880,delta=10)

    def test_resume_reuses_valid_outputs_and_renders_idempotently(self):
        first=self.execute()
        rendered=(self.run/'dialogue.md').read_bytes()
        hashes={p:r['sha256'] for p,r in first['media'].items()}
        models=FakeModels()
        second=self.execute(models,resume=True)
        self.assertEqual(models.calls,[])
        self.assertEqual(rendered,(self.run/'dialogue.md').read_bytes())
        self.assertEqual(hashes,{p:r['sha256'] for p,r in second['media'].items()})

    def test_failure_can_resume_without_redoing_completed_dependencies(self):
        with self.assertRaisesRegex(RuntimeError,'interruption'):
            self.execute(FakeModels(fail_prompt='variant'))
        failed=json.loads((self.run/'media_manifest.json').read_text())
        self.assertEqual(failed['status'],'failed')
        self.assertEqual((self.run/'dialogue.md').read_text(),self.original)
        canonical=self.run/'media/image_0002.png'
        timestamp=canonical.stat().st_mtime_ns
        models=FakeModels()
        self.assertEqual(self.execute(models,resume=True)['status'],'complete')
        self.assertEqual(canonical.stat().st_mtime_ns,timestamp)
        self.assertNotIn('reconstruct <Image 1>',[c['prompt'] for c in models.calls])

    def test_corrupt_output_is_regenerated(self):
        self.execute()
        (self.run/'media/image_0001.png').write_bytes(b'corrupt')
        models=FakeModels()
        self.execute(models,resume=True)
        self.assertIn('variant',[c['prompt'] for c in models.calls])

    def test_dry_run_has_no_outputs_or_model_calls(self):
        models=FakeModels()
        report=self.execute(models,dry_run=True)
        self.assertEqual(report['status'],'planned')
        self.assertEqual(models.calls,[])
        self.assertFalse((self.run/'media').exists())
        self.assertFalse((self.run/'media_manifest.json').exists())

    def test_h3_lora_switch_selects_steps_and_invalidates_resume(self):
        video = next(row for row in self.catalog if row.get('backend') == 'vdn_h3')
        video['backend'] = 'minimax_h3'
        self.persist()
        models = FakeModels()
        first = self.execute(models)
        self.assertFalse(first['settings']['video_turbo_lora'])
        self.assertEqual(first['settings']['video_steps_by_backend'], {'minimax_h3': 50})
        self.assertEqual(next(c for c in models.calls if c['backend'] == 'minimax_h3')['options']['steps'], 50)
        resumed = FakeModels()
        second = self.execute(resumed, resume=True, video_turbo_lora=True)
        self.assertTrue(second['settings']['video_turbo_lora'])
        self.assertEqual(next(c for c in resumed.calls if c['backend'] == 'minimax_h3')['options']['steps'], 6)
        report = self.execute(dry_run=True, video_turbo_lora=True, video_steps=8)
        self.assertEqual(report['settings']['video_steps_by_backend'], {'minimax_h3': 8})
        with self.assertRaisesRegex(ValueError, 'single pass'):
            self.execute(dry_run=True, video_two_pass=True, video_steps=8)

    def test_untracked_output_is_not_overwritten(self):
        (self.run/'media').mkdir()
        path=self.run/'media/video_0009.mp4'
        path.write_bytes(b'keep this file')
        with self.assertRaisesRegex(FileExistsError,'untracked'):
            self.execute()
        self.assertEqual(path.read_bytes(),b'keep this file')

    def test_missing_dependency_and_cycle_fail_before_execution(self):
        self.catalog[0]['reference_materials']=['[[image_9999]]'];self.catalog[0]['references']=[]
        self.persist()
        with self.assertRaisesRegex(ValueError,'Unknown generation dependency'):
            self.execute()
        self.catalog[0]['reference_materials']=['[[image_0001]]']
        self.persist()
        with self.assertRaisesRegex(ValueError,'Cyclic'):
            self.execute()
        self.assertFalse((self.run/'media').exists())

    def test_pending_prompts_and_misnumbered_references_fail_early(self):
        self.catalog[0]['status']='pending';self.persist()
        with self.assertRaisesRegex(ValueError,'rewriting'):
            self.execute()
        self.catalog[0].pop('status')
        self.catalog[0]['references'][0]['label']='<Image 2>';self.persist()
        with self.assertRaisesRegex(ValueError,'slot order'):
            self.execute()

    def test_virtual_source_requires_explicit_real_source_override(self):
        self.catalog[-1]['source_video']='fictional://example.mp4';self.persist()
        with self.assertRaisesRegex(ValueError,'real local file'):
            self.execute(dry_run=True)
        self.assertEqual(self.execute(source_video=self.target,dry_run=True)['status'],'planned')

    def test_existing_manifest_requires_resume(self):
        self.execute()
        original=(self.run/'media_manifest.json').read_bytes()
        with self.assertRaisesRegex(FileExistsError,'--resume'):
            self.execute()
        self.assertEqual(original,(self.run/'media_manifest.json').read_bytes())

    def test_optional_language_frame_preparation_is_reused_offline(self):
        from unittest.mock import patch
        from scripts.dialogue_generation.materialize import prepare_frames
        observation=json.loads((self.run/'observation.json').read_text())
        observation['assets']=[dict(id='cup',description='A white cup in the source video',frames=[.8])]
        write_json(self.run/'observation.json',observation)
        self.catalog[2]['asset_ids']=['cup'];self.persist()
        def selected(video, query, **kwargs):
            from PIL import Image
            self.assertIn('white cup',query)
            self.assertIsNotNone(kwargs['qwen'])
            self.assertEqual((kwargs['video_fps'],kwargs['coarse_frames']),(4.,32))
            self.assertEqual(kwargs['hint_timestamps'],[.8])
            Image.new('RGB',(64,64),'red').save(kwargs['frame_output'])
            return .8
        with patch('tools.find_video_frame.find_video_frame',side_effect=selected) as finder:
            report=prepare_frames(self.run,qwen=object(),progress=False)
            self.assertEqual(report['selections'],1)
            prepare_frames(self.run,qwen=object(),progress=False)
            self.assertEqual(finder.call_count,1)
        # Normal generation never calls the frame finder or vLLM.
        with patch('tools.find_video_frame.find_video_frame',side_effect=AssertionError('VLM must be off')):
            result=self.execute()
        ref=result['media']['[[image_0002]]']['references'][0]
        self.assertEqual(ref['frame_seconds'],.8)

    def test_stale_prepared_frame_fails_before_generation(self):
        from scripts.dialogue_generation.materialize import digest
        write_json(self.run/'frame_selections.json',dict(version=1,references={
            '[[image_0002]]:0':dict(fingerprint=digest('stale'),path='missing.png',sha256='bad')}))
        observation=json.loads((self.run/'observation.json').read_text())
        observation['assets']=[dict(id='cup',description='white cup')]
        write_json(self.run/'observation.json',observation)
        self.catalog[2]['asset_ids']=['cup'];self.persist()
        models=FakeModels()
        with self.assertRaisesRegex(ValueError,'rerun --prepare-frames'):
            self.execute(models)
        self.assertEqual(models.calls,[])

    def test_subsecond_video_reference_is_explicitly_padded(self):
        from cocreation import Reference
        from scripts.dialogue_generation.materialize import pad_short_references
        short=self.root/'short.mp4'
        movie(short,'red',440,seconds=.5)
        output=self.root/'inputs';output.mkdir()
        refs=pad_short_references([Reference('source',short,'video')],output)
        self.assertAlmostEqual(duration(refs[0].path),1,places=1)
        self.assertAlmostEqual(duration(short),.5,places=1)

    def test_forward_roots_need_no_guide_and_generated_video_can_feed_an_image(self):
        from unittest.mock import patch
        from PIL import Image
        models=FakeModels()
        self.catalog=[
            dict(placeholder='[[image_0001]]',kind='image',operation='text_generation',backend='flux',
                prompt='first asset',reference_materials=[]),
            dict(placeholder='[[video_0001]]',kind='video',operation='reference_generation',backend='vdn_h3',
                prompt='animate',seconds=5.,output_seconds=1.,reference_materials=['[[image_0001]]'],
                references=[dict(placeholder='[[image_0001]]',kind='image',label='<Picture 1>')]),
            dict(placeholder='[[image_0002]]',kind='image',operation='reference_generation',backend='flux',
                prompt='show asset <Image 1>',reference_materials=['[[video_0001]]'],
                references=[dict(placeholder='[[video_0001]]',kind='image',crop_query='the bag',label='<Image 1>')]),
            dict(placeholder='[[video_0002]]',kind='video',operation='copy',reference_materials=['[[video_0001]]'])]
        self.persist()
        self.original='[[image_0001]] [[video_0001]] [[image_0002]] [[video_0002]]'
        (self.run/'dialogue.md').write_text(self.original)
        write_json(self.run/'observation.json',dict(video='/does/not/exist/guide.mp4',
            video_metadata=dict(width=96,height=64),shots=[]))
        def crop(video,query,output):
            self.assertEqual(Path(video),self.run/'media/video_0001.mp4')
            self.assertEqual(query,'the bag')
            self.assertIn('vdn_h3',models.closed)
            Path(output).parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(64,64),'blue').save(output)
            return str(output)
        with patch('scripts.dialogue_generation.materialize.crop_generated_reference',side_effect=crop) as finder:
            result=self.execute(models)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['construction_order'][0],'[[image_0001]]')
        self.assertEqual(finder.call_count,1)
        self.assertEqual(models.calls[0]['references'],[])
        self.assertEqual([c['backend'] for c in models.calls],['flux','vdn_h3','flux'])
        self.assertIn('<img ',(self.run/'dialogue.md').read_text())

    def test_crop_failure_leaves_external_service_running(self):
        from unittest.mock import patch, Mock
        from scripts.dialogue_generation.materialize import crop_generated_reference
        response=Mock(status=200)
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        transport=Mock()
        with patch('urllib.request.urlopen',return_value=response), patch('cocreation.VLLMQwen',return_value=transport), \
             patch('tools.find_crop_in_video.find_crop_in_video',side_effect=RuntimeError('crop failure')), \
             patch('subprocess.Popen') as start, patch('os.killpg') as stop:
            with self.assertRaisesRegex(RuntimeError,'crop failure'):
                crop_generated_reference(self.a,'bag',self.root/'crop.png')
        start.assert_not_called()
        stop.assert_not_called()
        transport.close.assert_called_once()

    def test_crop_failure_releases_only_its_own_service(self):
        from unittest.mock import patch, Mock
        import signal
        from scripts.dialogue_generation.materialize import crop_generated_reference
        response=Mock(status=200)
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        process=Mock(pid=123456,poll=Mock(return_value=None))
        with patch('urllib.request.urlopen',side_effect=[OSError('offline'),response]), \
             patch('cocreation.VLLMQwen'), patch('subprocess.Popen',return_value=process), \
             patch('tools.find_crop_in_video.find_crop_in_video',side_effect=RuntimeError('crop failure')), \
             patch('os.killpg') as stop:
            with self.assertRaisesRegex(RuntimeError,'crop failure'):
                crop_generated_reference(self.a,'bag',self.root/'crop.png')
        stop.assert_called_once_with(123456,signal.SIGTERM)
        process.wait.assert_called_once()

    def test_default_model_uses_contract_checkpoint_and_existing_runtimes(self):
        from scripts.dialogue_generation.materialize import default_model
        from unittest.mock import patch
        with patch('scripts.dialogue_generation.materialize.cached_flux_snapshot',
                   return_value=self.root/'cached-revision'):
            flux=default_model('flux',None,self.root/'flux')
        self.assertEqual(flux.model_id,'black-forest-labs/FLUX.2-klein-4B')
        self.assertEqual(flux.runtime.revision,'cached-revision')
        self.assertTrue(flux.runtime.local_files_only)
        flux.close()
        video=default_model('vdn_h3',None,self.root/'vdn')
        self.assertFalse(video.runtime.auto_setup)
        self.assertTrue(video.runtime.local_files_only)
        video.close()
        for turbo in (False, True):
            h3=default_model('minimax_h3',None,self.root/'h3',video_turbo_lora=turbo)
            self.assertEqual(h3.runtime.turbo,turbo)
            self.assertTrue(h3.runtime.local_files_only)
            h3.close()

    def test_missing_checkpoint_fails_before_writing_outputs(self):
        from unittest.mock import patch
        with patch('scripts.dialogue_generation.materialize.cached_flux_snapshot',
                   side_effect=RuntimeError('FLUX checkpoint is not cached: test')):
            with self.assertRaisesRegex(RuntimeError,'checkpoint is not cached'):
                materialize(self.run,progress=False)
        self.assertFalse((self.run/'media_manifest.json').exists())
        self.assertFalse((self.run/'media').exists())

    def test_cached_snapshot_requires_all_sharded_weights(self):
        from unittest.mock import patch
        from scripts.dialogue_generation.materialize import cached_flux_snapshot
        snapshot=self.root/'snapshot'
        for name in ('scheduler','tokenizer','text_encoder','transformer','vae'):
            (snapshot/name).mkdir(parents=True)
        write_json(snapshot/'model_index.json',{})
        for name in ('text_encoder','vae'):
            (snapshot/name/'model.safetensors').write_bytes(b'weight')
        write_json(snapshot/'transformer/model.safetensors.index.json',
                   dict(weight_map={'layer':'missing.safetensors'}))
        with patch('huggingface_hub.snapshot_download',return_value=str(snapshot)):
            with self.assertRaisesRegex(RuntimeError,'missing.safetensors'):
                cached_flux_snapshot('test')
            (snapshot/'transformer/missing.safetensors').write_bytes(b'weight')
            self.assertEqual(cached_flux_snapshot('test'),snapshot)


if __name__=='__main__':
    unittest.main()
