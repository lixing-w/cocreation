import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from examples.benchmark import benchmark


class BenchmarkTests(unittest.TestCase):
    def exercise(self, selected):
        calls=[]; created={}; closed=[]
        class Fake:
            def __init__(self,name):
                self.name=name
                self.model_id='test/'+name
                created[name]=created.get(name,0)+1
            def __enter__(self):
                return self
            def __exit__(self,*exc):
                closed.append(self.name)
            def __call__(self,prompt,refs,**options):
                calls.append((self.name,id(self),prompt,tuple(refs),options))
                return {'status':'ok'}
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for cls,name in [('Qwen','qwen'),('Flux','flux'),('MiniMaxH3','minimax_h3'),('VDNH3','vdn_h3')]:
                stack.enter_context(patch('examples.benchmark.'+cls,side_effect=lambda name=name,**kw: Fake(name)))
            stack.enter_context(patch('examples.benchmark.time.perf_counter',side_effect=iter(range(100))))
            output=io.StringIO()
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            times=benchmark('image.jpg','video.mp4',Path(tmp),rounds=3,model=selected)
        active=['qwen','flux','vdn_h3'] if selected=='all' else [selected]
        self.assertEqual([c[0] for c in calls],active*3)
        self.assertEqual(created,dict.fromkeys(['qwen','flux','minimax_h3','vdn_h3'],1))
        self.assertEqual(set(closed),set(created))
        for name in active:
            records=[c for c in calls if c[0]==name]
            self.assertEqual(records[0],records[1]);self.assertEqual(records[1],records[2])
            self.assertEqual(times[name],[1,1,1])
            self.assertIn(f'{name}: 1.0s -> 1.0s -> 1.0s (3 rounds; dry-run only)',output.getvalue())

    def test_all_round_order_and_fixed_inputs(self):
        self.exercise('all')

    def test_single_model_rounds(self):
        self.exercise('flux')
