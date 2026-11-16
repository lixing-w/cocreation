"""Real ffmpeg round trip: rough observation cuts, exact frames and reordered audio."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.find_video_frame import frame_timestamps
from tools.video_editing import refine_shots,extract_shot,concatenate,probe


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class VideoEditing(unittest.TestCase):
    def test_observation_anchored_cuts_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            clips=[]
            for i,(color,freq) in enumerate([('red',440),('blue',660),('green',880)]):
                path=root/f'{i}.mp4'
                subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',f'color={color}:s=96x64:r=10:d=1',
                    '-f','lavfi','-i',f'sine=frequency={freq}:duration=1','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(path)],check=True)
                clips.append(path)
            video=concatenate(clips,root/'target.mp4')
            observation=dict(shots=[dict(id='S1',start=0,end=.94,description='red'),dict(id='S2',start=.94,end=2.06,description='blue'),dict(id='S3',start=2.06,end=3,description='green')])
            mapping=refine_shots(video,observation,window=.3)
            self.assertEqual(mapping['shot_order'],['S1','S2','S3'])
            self.assertEqual([b['frame_index'] for b in mapping['boundaries']],[10,20])
            rendered={sid:extract_shot(video,shot,root/f'{sid}.mp4') for sid,shot in mapping['shots'].items()}
            self.assertTrue(all(len(frame_timestamps(path))==10 for path in rendered.values()))
            edited=concatenate([rendered[s] for s in ('S3','S1','S2')],root/'reordered.mp4')
            self.assertEqual(len(frame_timestamps(edited)),30)
            self.assertTrue(any(s['codec_type']=='audio' for s in probe(edited)['streams']))
            # Decode center-frame colors after the full cut/assemble round trip.
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(edited),'-vf',"select='eq(n,5)+eq(n,15)+eq(n,25)',scale=1:1",'-fps_mode','vfr','-f','rawvideo','-pix_fmt','rgb24','-'])
            self.assertEqual([max(range(3),key=lambda c:raw[i+c]) for i in range(0,len(raw),3)],[1,0,2])
            # Audio stays aligned with its shot (frequency via zero crossings).
            audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(edited),'-f','s16le','-ac','1','-ar','8000','-'])
            import array
            samples=array.array('h',audio)
            frequencies=[]
            for second in (0,1,2):
                chunk=samples[int((second+.2)*8000):int((second+.8)*8000)]
                frequencies.append(sum(a<=0<b for a,b in zip(chunk,chunk[1:]))/.6)
            for actual,expected in zip(frequencies,(880,440,660)):
                self.assertAlmostEqual(actual,expected,delta=8)
            no_cut=dict(shots=[dict(id='A',start=0,end=.4,description='a'),dict(id='B',start=.4,end=3,description='b')])
            with self.assertRaisesRegex(ValueError,'No detected cut'):
                refine_shots(video,no_cut,window=.1)


if __name__=='__main__':unittest.main()
