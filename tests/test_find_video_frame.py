"""Best-match selection across batches, bounded boxes and original-resolution crops."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.find_crop_in_video import find_crop_in_video


@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
class CropTests(unittest.TestCase):
    def test_global_ranking_then_bbox_crop(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'input.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=160x96:r=8:d=2',
                '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            requests=[]
            class Model:
                def __call__(self,prompt,refs,**options):
                    requests.append((prompt,refs,options['system_prompt']))
                    if 'bbox_2d' in options['response_schema']['properties']:
                        return dict(text=json.dumps(dict(bbox_2d=[250,250,750,750])))
                    values=[int(Path(ref.path).stem.split('-')[-1]) for ref in refs]
                    return dict(text=json.dumps(dict(index=values.index(max(values)))))
                def close(self):
                    raise AssertionError('Injected model remains caller-owned')
            result=find_crop_in_video(video,'red subject',qwen=Model(),output_dir=root/'search',
                crop_output=root/'crop.png',coarse_frames=8,refine_frames=4,batch_frames=4,video_fps=4)
            self.assertEqual(result['source_frame_index'],15)
            self.assertEqual(result['bbox_pixels'],[40,24,120,72])
            with Image.open(root/'crop.png') as crop:
                self.assertEqual(crop.size,(80,48))
                self.assertGreater(crop.getpixel((5,5))[0],240)
            self.assertGreaterEqual(len([r for r in result['rounds'] if r['stage']=='coarse']),2)
            global_round=next(r for r in result['rounds'] if r['stage']=='coarse-global')
            self.assertEqual(global_round['selected_source_index'],15)
            self.assertIn('Always choose the best available',requests[0][2])
            self.assertNotIn('red subject',requests[0][2])
            self.assertIn('red subject',requests[0][0])
            self.assertIn('Coordinates are normalized',requests[-1][2])
            with self.assertRaises(FileExistsError):
                find_crop_in_video(video,'red subject',qwen=Model(),crop_output=root/'crop.png')


if __name__=='__main__':
    unittest.main()
