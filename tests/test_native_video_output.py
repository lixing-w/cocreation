"""Native-video mode preserves both requested canvases and the original MP4."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.output_geometry import geometry
from scripts.dialogue_generation.materialize import materialize
from tools.video_editing import probe


class NativeGeometryTests(unittest.TestCase):
    def test_video_presets_keep_the_requested_native_canvas(self):
        for preset, expected in [(None, (1344,768)), ('16:9',(1344,768)),
                                 ('9:16',(768,1344)), ('1:1',(1344,768)),
                                 ('square_assets_portrait_video',(768,1344))]:
            with self.subTest(preset=preset):
                state = {} if preset is None else dict(aspect_ratio=preset)
                self.assertEqual(geometry(state,'video',video=(1344,768),native_video_output=True),
                                 (expected,expected))

    def test_native_default_orients_landscape_even_with_portrait_model_dimensions(self):
        self.assertEqual(geometry({}, 'video', video=(768, 1344), native_video_output=True),
                         ((1344, 768), (1344, 768)))
        self.assertEqual(geometry({'aspect_ratio': 'portrait'}, 'video', video=(768, 1344),
                                  native_video_output=True), ((768, 1344), (768, 1344)))

    def test_image_geometry_is_unaffected(self):
        self.assertEqual(geometry(dict(aspect_ratio='1:1'),'image',video=(1344,768),native_video_output=True),
                         ((1024,1024),(1024,1024)))


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
class NativeMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(placeholder='[[video_0001]]',kind='video',operation='text_generation',
                         backend='minimax_h3',prompt='A blue wall.',dependencies=[],references=[],
                         render_state=dict(aspect_ratio='16:9')),
                     dict(placeholder='[[video_0002]]',kind='video',operation='video_edit',
                          backend='minimax_h3',prompt='Retain the wall in <Video 1>.',
                          dependencies=['[[video_0001]]'],
                          references=[dict(placeholder='[[video_0001]]',kind='video',label='<Video 1>')],
                          render_state=dict(aspect_ratio='9:16'))]
        (self.root/'placeholders.json').write_text(json.dumps(self.rows))
        (self.root/'dialogue.md').write_text('[[video_0001]] [[video_0002]]')

    def test_original_bytes_audio_duration_and_both_resolutions_are_preserved(self):
        outputs, calls = [], []
        def factory(backend, model_id, output_dir):
            class Model:
                def __init__(self):
                    self.output_dir=Path(output_dir)
                def __call__(self, prompt, references, **options):
                    calls.append(options)
                    target=Path(self.output_dir)/'video.mp4'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                        f"color=blue:s={options['width']}x{options['height']}:r=24:d={107/24}",
                        '-f','lavfi','-i',f'sine=frequency=660:sample_rate=32000:duration={107/24}',
                        '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p',
                        '-c:a','aac','-ac','2','-shortest',str(target)],check=True)
                    outputs.append(hashlib.sha256(target.read_bytes()).hexdigest())
                    return dict(status='ok',path=str(target))
                def close(self): pass
            return Model()
        with patch('scripts.dialogue_generation.materialize.finish_video',side_effect=AssertionError('Native MP4 must bypass export conversion')):
            result=materialize(self.root,video_width=1344,video_height=768,native_video_output=True,
                               model_factory=factory,progress=False,video_steps=40)
        self.assertEqual(result['status'],'complete')
        for index, expected in enumerate(((1344,768),(768,1344))):
            key=f'[[video_{index+1:04d}]]'
            record=result['media'][key]
            target=self.root/record['path']
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),outputs[index])
            info=probe(target)
            stream=next(s for s in info['streams'] if s['codec_type']=='video')
            self.assertEqual((stream['width'],stream['height']),expected)
            self.assertEqual(stream['nb_frames'],'107')
            self.assertTrue(any(s['codec_type']=='audio' for s in info['streams']))
            self.assertAlmostEqual(float(stream['duration']),107/24,places=2)
            self.assertEqual(record['video_output_mode'],'native')
            self.assertEqual((calls[index]['width'],calls[index]['height']),expected)
            self.assertEqual(calls[index]['steps'],40)

    def test_export_size_overrides_are_rejected_in_native_mode(self):
        with self.assertRaisesRegex(ValueError,'output size overrides'):
            materialize(self.root,video_width=1344,video_height=768,native_video_output=True,
                        output_width=1280,output_height=720,dry_run=True)


if __name__ == '__main__':
    unittest.main()
