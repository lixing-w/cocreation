"""Natural-language pool sampling and complete offline pipeline regression tests."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch
from cocreation import Reference
from tools._attribute_protocol import (sample_selection, sample_count, prune_references,
    validate_sources, media_catalog, source_description)
from scripts.cpg import cpg_reverse as cpg


class ProtocolTests(unittest.TestCase):
    def test_zero_selection_works_even_with_empty_pool(self):
        self.assertEqual(sample_selection([],0,random.Random(0)),('',[]))
        with self.assertRaises(ValueError): sample_selection([],-1,random.Random(0))

    def test_decreasing_count_distribution_and_seed(self):
        from collections import Counter
        rng=random.Random(42)
        sequence=[sample_count((0,3),rng) for _ in range(10000)]
        counts=Counter(sequence)
        for k,p in enumerate((.4,.3,.2,.1)):
            self.assertAlmostEqual(counts[k]/10000,p,delta=.015)
        again=random.Random(42)
        self.assertEqual(sequence[:20],[sample_count((0,3),again) for _ in range(20)])
        self.assertEqual(sample_count((0,0),rng),0)
        self.assertEqual(sample_count((3,3),rng),3)
        with self.assertRaises(ValueError): sample_count((-1,3),rng)

    def test_sampling_preserves_actual_carrier_and_seed(self):
        refs=[Reference(f'artifact_{i}',f'/tmp/{i}.png','image') for i in range(4)]
        pool=[(f'Subject {i} stands.\nSubject {i} wears a hat.',ref) for i,ref in enumerate(refs)]
        first=sample_selection(pool,3,random.Random(4),max_parents=2)
        self.assertEqual(first,sample_selection(pool,3,random.Random(4),max_parents=2))
        text,used=first
        self.assertEqual(len(text.splitlines()),3)
        self.assertLessEqual(len(used),2)
        for ref,label in ((ref,media_catalog(used)[ref.artifact_id]) for ref in used):
            self.assertTrue(all('Subject '+ref.artifact_id[-1] in line
                                for line in text.splitlines() if label in line))
        self.assertIsNone(sample_selection(pool,9,random.Random(1)))
        self.assertTrue(all('(from' not in text for text,ref in pool))

    def test_pruning_renumbers_simultaneously_without_changing_carrier(self):
        refs=[Reference('a','/a','image'),Reference('v','/v','video'),
              Reference('b','/b','image'),Reference('c','/c','image')]
        text='B wears a hat. (from <Image 2>)\nC wears silk. (from <Image 3>)\nWater ripples. (from <Video 1>)'
        text,used=prune_references(text,refs)
        self.assertEqual([r.artifact_id for r in used],['v','b','c'])
        self.assertIn('B wears a hat. (from <Image 1>)',text)
        self.assertIn('C wears silk. (from <Image 2>)',text)
        self.assertEqual(source_description(text,'<Video 1>'),'Water ripples.')
        self.assertEqual(prune_references('A new world.',refs),('A new world.',[]))
        with self.assertRaises(ValueError): validate_sources('A hat. (from <Image 4>)',refs)

    def test_identical_descriptions_do_not_deduplicate_artifacts(self):
        refs=[Reference('a','/a','image'),Reference('b','/b','image')]
        text,used=sample_selection([('A hat.',ref) for ref in refs],2,random.Random(0))
        self.assertEqual(len(used),2)
        self.assertEqual(len(text.splitlines()),2)


class Model:
    def __init__(self, callback): self.callback=callback; self.calls=[]; self.closed=False
    def __enter__(self): return self
    def __exit__(self,*args): self.closed=True
    def __call__(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return self.callback(*args,**kwargs)


class LoopTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        for name in ('source.png','source.mp4','generated.png','generated.mp4'):
            (self.root/name).touch()
        self.manifest=self.root/'roots.json'
        self.roots=[dict(artifact_id='artifact_0',kind='image',path='source.png')]
        self.manifest.write_text(json.dumps(self.roots))
        self.rewrite='A sculptural hat. (from <Image 1>)\nA satin arch floats above pale sand.'
        self.verified='A black sculptural hat rests on pale sand.\nA satin arch floats overhead.'
        self.events=[]
        def respond(prompt,refs,**options):
            system=options['system_prompt']
            self.events.append((system,prompt,list(refs)))
            if system.startswith('You are extractor'):
                return {'text':'A woman wears a sculptural hat.\nThe studio is white.'}
            if system.startswith('You are attribute_rewriter'):
                return {'text':self.rewrite}
            if system.startswith('You are attribute_verifier'):
                return {'text':self.verified}
            if system.startswith('You are the FLUX'):
                from tools._attribute_protocol import SOURCE
                labels=[]
                for i,ref in enumerate(refs,1): labels.append(f'<Image {i}>')
                return {'text':'An elegant portrait '+('with a hat from '+' and '.join(labels) if labels else 'of a satin arch')+'.'}
            if system.startswith('You are the VDN'):
                return {'text': 'subject_definitions:\nA hat from <Picture 1>.\nsummary:\nA hat floats.\n'
                    'retention_analysis:\nPreserve the hat from <Picture 1>.\ndetailed_description:\n'
                    '[Shot 1] A hat floats over sand.\noverall_soundscape:\nSoft wind.\nnon_diegetic_music:\nN/A'}
            self.fail(system)
        self.qwen=Model(respond)
        self.flux=Model(lambda *a,**kw:dict(kind='image',path=str(self.root/'generated.png')))
        self.vdn=Model(lambda *a,**kw:dict(kind='video',path=str(self.root/'generated.mp4')))
        for name,model in [('Qwen',self.qwen),('Flux',self.flux),('VDNH3',self.vdn)]:
            self.enterContext(patch.object(cpg,name,return_value=model))

    def run_loop(self,**kwargs):
        return cpg.run(self.manifest,self.root/'run',**dict(
            dict(attempts=1,keep=(1,1),p_video=0,seed=0,debug=False),**kwargs))

    def test_real_tools_complete_loop_and_recycle_verified_artifact(self):
        graph=self.run_loop(attempts=2)
        self.assertEqual(graph['version'],2)
        self.assertEqual([a['status'] for a in graph['attempts']],['accepted','accepted'])
        self.assertEqual(len(graph['artifacts']),3)
        self.assertTrue(all('provenance' not in n for n in graph['artifacts']))
        self.assertEqual(graph['artifacts'][-1]['description'],self.verified)
        verifier_events=[e for e in self.events if e[0].startswith('You are attribute_verifier')]
        self.assertEqual(len(verifier_events),2)
        for system,user,refs in verifier_events:
            self.assertEqual(len(refs),1)
            self.assertEqual(refs[0].kind,'image')
            attempt_index=verifier_events.index((system,user,refs))
            plan=(self.root/f'run/attempt_{attempt_index:04d}/plan.txt').read_text()
            self.assertIn(plan,user)
        self.assertTrue(self.qwen.closed and self.flux.closed and self.vdn.closed)
        self.assertTrue((self.root/'run/attempt_0000/plan.txt').is_file())
        self.assertFalse((self.root/'run/attempt_0000/inheritance.json').exists())
        # Force sampling just the generated pair to prove its text binds to itself.
        n=graph['artifacts'][-1]
        text,refs=sample_selection([(n['description'],Reference(n['artifact_id'],n['path'],n['kind']))],1,random.Random(0))
        self.assertEqual(refs[0].artifact_id,n['artifact_id'])
        self.assertNotIn('artifact_0',text)

    def test_pruned_reference_is_renumbered_before_prompter_and_verifier(self):
        self.roots.append(dict(artifact_id='artifact_1',kind='image',path='source.png'))
        self.manifest.write_text(json.dumps(self.roots))
        self.rewrite='A sculptural hat. (from <Image 2>)'
        graph=self.run_loop(keep=(4,4))
        attempt=graph['attempts'][0]
        self.assertEqual(attempt['status'],'accepted')
        retained=attempt['sampled_parents'][1]
        self.assertEqual(graph['artifacts'][-1]['parents'],[retained])
        self.assertEqual(self.flux.calls[0][0][1][0].artifact_id,retained)
        for system,user,refs in self.events:
            if system.startswith(('You are the FLUX','You are attribute_verifier')):
                self.assertIn('A sculptural hat. (from <Image 1>)',user)
                self.assertNotIn('(from <Image 2>)',user)

    def test_source_free_rewrite_sends_no_media_to_generator(self):
        self.rewrite='A satin arch floats above pale sand.'
        graph=self.run_loop(keep=(0,0),p_video=1)
        self.assertEqual(graph['attempts'][0]['status'],'accepted')
        self.assertEqual(self.flux.calls[0][0][1],[])
        self.assertEqual(len(self.vdn.calls),0)
        self.assertEqual(graph['attempts'][0]['sampled_parents'],[])
        self.assertEqual(graph['attempts'][0]['mode'],'new_identity')
        self.assertEqual(graph['artifacts'][-1]['parents'],[])
        self.assertIn(self.rewrite,next(e[1] for e in self.events if e[0].startswith('You are attribute_verifier')))

    def test_video_branch_uses_h3_and_same_rewritten_description(self):
        graph=self.run_loop(p_video=1)
        self.assertEqual(graph['attempts'][0]['status'],'accepted')
        self.assertEqual(len(self.vdn.calls),1)
        self.assertEqual(len(self.flux.calls),0)
        self.assertEqual(graph['artifacts'][-1]['kind'],'video')
        self.assertIn((self.root/'run/attempt_0000/plan.txt').read_text(),
                      next(e[1] for e in self.events if e[0].startswith('You are the VDN')))
        self.assertFalse(any(e[0].startswith('You are attribute_rewriter') for e in self.events))

    def test_failures_never_enter_pool(self):
        self.rewrite='A hat. (from <Image 9>)'
        graph=self.run_loop(keep=(0,0))
        self.assertEqual(len(graph['artifacts']),1)
        self.assertEqual(graph['attempts'][0]['stage'],'attribute_rewriter')
        self.assertEqual(len(self.flux.calls),0)

    def test_verification_failure_never_enters_pool(self):
        self.verified='A hat. (from <Image 1>)'
        graph=self.run_loop()
        self.assertEqual(len(graph['artifacts']),1)
        self.assertEqual(graph['attempts'][0]['stage'],'attribute_verifier')

    def test_operational_failure_checkpoints_and_closes_models(self):
        self.flux.callback=Mock(side_effect=RuntimeError('generator unavailable'))
        with self.assertRaises(RuntimeError): self.run_loop()
        graph=json.loads((self.root/'run/graph.json').read_text())
        self.assertEqual(graph['attempts'][0]['status'],'error')
        self.assertEqual(graph['attempts'][0]['stage'],'generation')
        self.assertEqual(len(graph['artifacts']),1)
        self.assertTrue(self.qwen.closed and self.flux.closed and self.vdn.closed)

    def test_video_frame_query_is_prose_and_slot_is_preserved(self):
        self.roots[0].update(kind='video',path='source.mp4')
        self.manifest.write_text(json.dumps(self.roots))
        self.rewrite='A sculptural hat. (from <Video 1>)'
        def frame(path,description,**options):
            self.assertIn(description,('A woman wears a sculptural hat.','The studio is white.'))
            options['frame_output'].touch()
            return 0.4
        with patch.object(cpg,'find_video_frame',side_effect=frame): graph=self.run_loop()
        ref=self.flux.calls[0][0][1][0]
        self.assertEqual(ref.kind,'image')
        self.assertEqual(ref.artifact_id,'artifact_0')
        self.assertEqual(graph['attempts'][0]['frames'][0]['timestamp_seconds'],0.4)
        self.assertEqual(graph['artifacts'][-1]['parents'],['artifact_0'])

    def test_no_matching_video_frame_skips_generation(self):
        self.roots[0].update(kind='video',path='source.mp4')
        self.manifest.write_text(json.dumps(self.roots))
        self.rewrite='A sculptural hat. (from <Video 1>)'
        with patch.object(cpg,'find_video_frame',return_value=None): graph=self.run_loop()
        self.assertEqual(graph['attempts'][0]['stage'],'frame_search')
        self.assertEqual(len(self.flux.calls),0)

    def test_default_count_policy_recorded_and_zero_exploration_rejoins_pool(self):
        self.rewrite='An ivory cape has a sculptural collar.'
        graph=cpg.run(self.manifest,self.root/'run',attempts=0,debug=False)
        self.assertEqual(graph['config']['keep'],[0,3])
        self.assertEqual(graph['config']['sampling_weights'],[4,3,2,1])
        self.assertEqual(graph['config']['zero_attribute_kind'],'image')

    def test_supplied_text_file_and_legacy_migration_error(self):
        (self.root/'description.txt').write_text('A white studio.')
        self.roots[0]['description_file']='description.txt'
        self.manifest.write_text(json.dumps(self.roots))
        self.assertEqual(cpg.read_roots(self.manifest)[0]['description'],'A white studio.')
        graph=self.run_loop(attempts=0)
        self.assertEqual(len(self.events),0)
        self.roots[0]['observation']={'attributes':{}}
        self.manifest.write_text(json.dumps(self.roots))
        with self.assertRaisesRegex(ValueError,'Legacy JSON'): cpg.read_roots(self.manifest)


if __name__ == '__main__': unittest.main()
