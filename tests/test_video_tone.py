"""Root lineage plus real-media brightness, timing and resume integration."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from cocreation.video_tone import brightness_anchor, joint_curve, render_curve, sample_planes
from scripts.dialogue_generation.materialize import materialize, file_hash


def token(n):
    return f'[[video_{n:04d}]]'


def task(n, op='reference_generation', refs=()):
    return dict(placeholder=token(n), kind='video', operation=op, backend='vdn_h3',
                prompt=f'clip {n}', status='ready', output_seconds=2.,
                dependencies=[token(i) for i in refs], references=[
                    dict(placeholder=token(i), kind='video', label=f'<Video {j+1}>') for j, i in enumerate(refs)])


class LineageTests(unittest.TestCase):
    def test_original_root_through_edits_and_copy_but_new_generation_starts_chain(self):
        rows = [task(1), task(2,'video_edit',[1]), task(3,'video_edit',[2]),
                dict(placeholder=token(4),operation='copy',dependencies=[token(3)]),
                task(5,'video_edit',[4]),task(6,refs=[1,3]),task(7,'video_edit',[6])]
        tasks = {r['placeholder']:r for r in rows}
        for n in (2,3,5):
            self.assertEqual(brightness_anchor(tasks[token(n)],tasks)['placeholder'], token(1))
        self.assertIsNone(brightness_anchor(tasks[token(6)],tasks))
        self.assertEqual(brightness_anchor(tasks[token(7)],tasks)['placeholder'],token(6))

    def test_ambiguous_edit_fails_but_multiple_generation_refs_are_valid(self):
        with self.assertRaisesRegex(ValueError,'exactly one base video'):
            brightness_anchor(task(3,'video_edit',[1,2]),{token(1):task(1),token(2):task(2)})
        with self.assertRaisesRegex(ValueError,'exactly one base video'):
            brightness_anchor(task(3,'video_edit'),{})

    def test_legacy_shot_selector_is_inherited(self):
        v2 = task(2,'video_edit',[1]);v2['references'][0]['shot_id']='S2'
        v3 = task(3,'video_edit',[2])
        tasks={token(1):task(1),token(2):v2,token(3):v3}
        self.assertEqual(brightness_anchor(v3,tasks),dict(placeholder=token(1),kind='video',shot_id='S2'))

    def test_curve_keeps_endpoints_and_validates_all_tone_bands(self):
        y=np.broadcast_to(np.linspace(16,235,96),(24,64,96)).astype(np.float32)
        anchor=np.stack((y,np.full_like(y,110),np.full_like(y,146)),axis=1)
        target=anchor.copy();target[:,0]=16+(y-16)*.8
        # sample_planes normalizes luma to 0..255.
        anchor[:,0]=(anchor[:,0]-16)*255/219
        target[:,0]=(target[:,0]-16)*255/219
        lut, report=joint_curve(anchor,target)
        self.assertIsNotNone(lut)
        self.assertTrue(np.all(np.diff(lut.astype(int))>=0))
        self.assertEqual((int(lut[16]),int(lut[235])),(16,235))
        self.assertLess(report['balanced_mae_after'],report['balanced_mae_before']*.1)
        changed=target.copy();changed[:,1:]+=80
        lut, report=joint_curve(anchor,changed)
        self.assertIsNone(lut)
        self.assertEqual(report['reason'],'insufficient_stable_content')


class GradientModels:
    def __init__(self):
        self.calls=[]
    def __call__(self,backend,model_id,output_dir):
        owner=self
        class Model:
            def __init__(self):self.output_dir=Path(output_dir)
            def close(self):pass
            def __call__(self,prompt,references,**options):
                owner.calls.append(dict(prompt=prompt, reference_means=[
                    float(sample_planes(r.path)[0][:,0].mean()) for r in references]))
                self.output_dir.mkdir(parents=True,exist_ok=True)
                path=self.output_dir/'generated.mp4'
                gain=1 if prompt=='clip 1' else .8 if prompt=='clip 2' else .65
                subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                    f"nullsrc=s=96x64:r=12:d=2,geq=lum='16+{gain}*219*X/W':cb=110:cr=146",
                    '-f','lavfi','-i','sine=frequency=440:duration=2',
                    '-c:v','libx264','-crf','10','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(path)],check=True)
                return dict(status='ok',path=str(path))
        return Model()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class VideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)
        self.rows=[task(1),task(2,'video_edit',[1]),task(3,'video_edit',[2]),task(4,refs=[1,3])]
        (self.run/'placeholders.json').write_text(json.dumps(self.rows))
        (self.run/'dialogue.md').write_text(' '.join(token(i) for i in range(1,5)))
    def execute(self,models,**kwargs):
        return materialize(self.run,model_factory=models,output_width=96,output_height=64,progress=False,**kwargs)
    def test_chain_uses_original_and_corrected_parent_is_next_model_input(self):
        models=GradientModels();report=self.execute(models)
        media=report['media']
        root_mean=float(sample_planes(self.run/media[token(1)]['path'])[0][:,0].mean())
        for n in (2,3):
            match=media[token(n)]['brightness_matching']
            self.assertEqual(match['anchor']['placeholder'],token(1))
            self.assertEqual(match['anchor_sha256'],media[token(1)]['sha256'])
            self.assertEqual(match['status'],'matched')
            output_mean=float(sample_planes(self.run/media[token(n)]['path'])[0][:,0].mean())
            self.assertAlmostEqual(output_mean,root_mean,delta=2.)
        third=next(c for c in models.calls if c['prompt']=='clip 3')
        self.assertAlmostEqual(third['reference_means'][0],root_mean,delta=2.)
        self.assertNotIn('brightness_matching',media[token(4)])
        resumed=GradientModels();self.execute(resumed,resume=True)
        self.assertEqual(resumed.calls,[])
    def test_enabling_on_resume_reuses_roots_and_changes_only_edits_and_dependents(self):
        first=self.execute(GradientModels(),match_video_brightness=False)
        root_hash=first['media'][token(1)]['sha256']
        models=GradientModels();self.execute(models,resume=True)
        self.assertNotIn('clip 1',[c['prompt'] for c in models.calls])
        self.assertIn('clip 2',[c['prompt'] for c in models.calls])
        self.assertEqual(file_hash(self.run/'media/video_0001.mp4'),root_hash)
    def test_dry_run_reports_anchors_and_rejects_multiple_edit_bases(self):
        models=GradientModels();report=self.execute(models,dry_run=True)
        self.assertEqual(models.calls,[])
        self.assertTrue(report['video_brightness_matching']['enabled'])
        self.assertEqual(report['video_brightness_matching']['anchors'][token(3)][0]['placeholder'],token(1))
        self.rows[2]=task(3,'video_edit',[1,2])
        (self.run/'placeholders.json').write_text(json.dumps(self.rows))
        with self.assertRaisesRegex(ValueError,'exactly one base video'):
            self.execute(models,dry_run=True)
    def test_render_preserves_vfr_timestamps_audio_chroma_and_color_metadata(self):
        source=self.run/'vfr.mp4';dest=self.run/'matched.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
            'testsrc2=s=96x64:r=12:d=2','-f','lavfi','-i','sine=frequency=880:duration=2',
            '-vf',"select='not(eq(mod(n,4),1))'",'-fps_mode','vfr',
            '-c:v','libx264','-crf','0','-c:a','aac','-color_range','tv','-colorspace','bt709',
            '-color_primaries','bt709','-color_trc','bt709',str(source)],check=True)
        lut=np.arange(256,dtype=np.uint8);lut[17:235]=np.minimum(234,lut[17:235].astype(int)+5)
        render_curve(source,dest,lut)
        def packets(path,selector):
            data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams',selector,
                '-show_packets','-show_data_hash','sha256','-of','json',str(path)]))
            return data['packets']
        self.assertEqual(sorted(p['pts_time'] for p in packets(source,'v:0')),
                         sorted(p['pts_time'] for p in packets(dest,'v:0')))
        self.assertEqual(len(packets(source,'v:0')),len(packets(dest,'v:0')))
        self.assertEqual([p['data_hash'] for p in packets(source,'a:0')],[p['data_hash'] for p in packets(dest,'a:0')])
        # U/V changes are limited to encoder loss, without a colorspace conversion.
        a,_=sample_planes(source);b,_=sample_planes(dest)
        self.assertLess(float(np.abs(a[:,1:]-b[:,1:]).mean()),1.5)


if __name__=='__main__':unittest.main()
