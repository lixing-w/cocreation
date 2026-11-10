import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from tools.find_video_frame import find_video_frame, frame_timestamps, parse_selection, sample_indices, parse_location


class FrameTests(unittest.TestCase):
    def test_invalid_model_answers_are_errors(self):
        for text in ('{"frame_id":true}', '{"frame_id":-1}', '{"frame_id":3}',
                     '{"timestamp":1.2}', 'not found', '{"frame_id":1.0}', '1.0', '-1', '3', 'true'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_selection(text, 3)
        self.assertIsNone(parse_selection('NOT_FOUND', 3))

    def test_invalid_video_locations(self):
        for value in (True, -1, 100, float('nan'), float('inf'), '1.5 seconds'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_location(str(value), 2.0)
        self.assertIsNone(parse_location('NOT_FOUND', 2.0))

    def test_sampling_uses_real_vfr_timestamps(self):
        times=[0., .01, .04, .9, 1., 1.5, 3.]
        indices=sample_indices(times, 4)
        self.assertEqual(indices, [0, 4, 5, 6])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
    def test_real_video_roundtrip_and_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); video=root/'source.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                'testsrc=size=64x64:rate=10:duration=2','-pix_fmt','yuv420p',str(video)],check=True)
            times=frame_timestamps(video)
            self.assertEqual(len(times),20)
            expected=root/'expected.png'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-vf',
                'select=eq(n\\,6)','-vsync','0','-frames:v','1',str(expected)],check=True)
            # Coarse chooses frame 6, fine chooses the same source frame.
            def choose(prompt, references, **options):
                if references[0].kind == 'video':
                    self.assertEqual(len(references), 1)
                    self.assertEqual(options['video_fps'], 2.0)
                    self.assertEqual(options['video_max_frames'], 4)
                    return {'text': '0.6'}
                self.assertNotIn('source_frame_index', prompt)
                self.assertNotIn('timestamp_seconds', prompt)
                wanted=next(i for i, ref in enumerate(references)
                            if Path(ref.path).read_bytes() == expected.read_bytes())
                self.assertTrue(all(Path(r.path).is_file() for r in references))
                return {'text':str(wanted)}
            qwen=Mock(side_effect=choose)
            export=root/'selected.png'
            result=find_video_frame(video,'A clear test pattern',qwen=qwen,
                coarse_frames=4,refine_frames=6,output_dir=root/'runs',frame_output=export)
            self.assertAlmostEqual(result,.6)
            self.assertEqual(qwen.call_count,2)
            qwen.close.assert_not_called()
            self.assertEqual(export.read_bytes(),expected.read_bytes())
            qwen.reset_mock(); qwen.side_effect=None; qwen.return_value={'text':'NOT_FOUND'}
            result=find_video_frame(video,'missing subject',qwen=qwen,
                output_dir=root/'runs',frame_output=root/'missing.png')
            self.assertIsNone(result)
            self.assertFalse((root/'missing.png').exists())
            self.assertEqual(qwen.call_count,1)
            with patch('tools.find_video_frame.Qwen', return_value=qwen):
                find_video_frame(video,'missing',output_dir=root/'runs')
            qwen.close.assert_called_once()
