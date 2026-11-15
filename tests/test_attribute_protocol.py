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

    def test_each_carrier_has_at_most_two_lines(self):
        refs=[Reference('a','/a','image'),Reference('b','/b','image')]
        pool=[('A silver gown.\nThe fabric shines.\nThe hem curves.',refs[0]),
              ('A garden.\nA stone wall.',refs[1])]
        self.assertIsNone(sample_selection(pool[:1],3,random.Random(0)))
        accepted=0
        for seed in range(100):
            result=sample_selection(pool,3,random.Random(seed))
            if result is None: continue
            text,used=result
            accepted+=1
            for label in media_catalog(used).values():
                self.assertLessEqual(text.count('(from '+label+')'),2)
            self.assertEqual(len(used),2)
        self.assertGreater(accepted,0)

    def test_line_combination_dedup_ignores_order_and_temporary_labels(self):
        refs=[Reference('a','/a','image'),Reference('b','/b','image')]
        pool=[('A hat.',refs[0]),('A chair.',refs[1])]
        seen=set(); rows=[]
        rng=Mock()
        rng.sample.side_effect=lambda candidates,k:list(candidates)
        self.assertIsNotNone(sample_selection(pool,2,rng,seen_combinations=seen,selected_lines=rows))
        self.assertEqual(rows,[dict(artifact_id='a',line_number=1),dict(artifact_id='b',line_number=1)])
        rng.sample.side_effect=lambda candidates,k:list(reversed(candidates))
        self.assertIsNone(sample_selection(pool,2,rng,seen_combinations=seen,selected_lines=rows))
        self.assertEqual(rows,[])
        self.assertEqual(len(seen),1)

    def test_same_text_on_different_lines_is_distinct_and_zero_is_repeatable(self):
        ref=Reference('a','/a','image'); seen=set(); rows=[]
        pool=[('A hat.\n\nA hat.',ref)]
        rng=Mock(); rng.sample.side_effect=lambda candidates,k:candidates[:1]
        self.assertIsNotNone(sample_selection(pool,1,rng,seen_combinations=seen,selected_lines=rows))
        self.assertEqual(rows[0]['line_number'],1)
        rng.sample.side_effect=lambda candidates,k:candidates[-1:]
        self.assertIsNotNone(sample_selection(pool,1,rng,seen_combinations=seen,selected_lines=rows))
        self.assertEqual(rows[0]['line_number'],3)
        for _ in range(2):
            self.assertEqual(sample_selection(pool,0,rng,seen_combinations=seen),('',[]))
        self.assertEqual(len(seen),2)

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


if __name__ == "__main__": unittest.main()
