"""Natural-language observation tools and shared VLM lifecycle."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cocreation import Reference
from tools.extractor import extractor
from tools.extract_attribute_nodes import extract_attribute_nodes
from tools.attribute_rewriter import attribute_rewriter
from tools.attribute_verifier import attribute_verifier
from tools.extract_attribute_nodes_prompts import extraction_system_prompt
from tools.attribute_rewriter_prompts import SYSTEM as REWRITE
from tools.attribute_verifier_prompts import SYSTEM as VERIFY


class ObservationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        path = self.root/'image.png'; path.touch()
        self.ref = Reference('artifact_23', path, 'image')
        self.description = 'A woman stands in a studio.\nThe woman wears a white hat.'

    def test_extractor_returns_prose_with_no_source_annotations(self):
        model = Mock(return_value={'text': self.description})
        self.assertEqual(extract_attribute_nodes(self.ref, qwen=model,
                         output_dir=self.root/'out'), self.description)
        user, refs = model.call_args.args
        self.assertEqual(refs, [self.ref])
        self.assertNotIn('"artifact_id"', user)
        self.assertIn('{{artifact:artifact_23}}', user)
        self.assertIn('You are extractor', model.call_args.kwargs['system_prompt'])
        self.assertTrue((self.root/'out/input.txt').is_file())
        self.assertEqual((self.root/'out/response.txt').read_text(), self.description)
        model.close.assert_not_called()

    def test_invalid_observations_and_owned_lifecycle(self):
        for output in ('{"attributes":{}}', '```text\nA woman.\n```',
                       'A woman. (from <Image 1>)', '', 'NO_OBSERVATION'):
            model = Mock(return_value={'text':output})
            with self.subTest(output=output), patch('tools._attribute_vlm.Qwen', return_value=model):
                self.assertIsNone(extractor(self.ref, output_dir=self.root/'out'))
                model.close.assert_called_once()
        model = Mock(side_effect=RuntimeError('inference failure'))
        with patch('tools._attribute_vlm.Qwen', return_value=model):
            with self.assertRaises(RuntimeError):
                extractor(self.ref, output_dir=self.root/'out')
        model.close.assert_called_once()

    def test_same_source_is_exact_passthrough_without_inference(self):
        sampled = 'The woman wears a hat. (from <Image 1>)\nA white studio. (from <Image 1>)'
        model = Mock(side_effect=AssertionError('Must not call model'))
        self.assertEqual(attribute_rewriter(sampled, [self.ref], kind='image', qwen=model,
                         output_dir=self.root/'rewrite'), sampled)
        model.assert_not_called()
        self.assertEqual((self.root/'rewrite/response.txt').read_text(), sampled)

    def test_zero_attributes_creates_identity_without_media(self):
        output = 'A sculptural wool cape has a crescent collar.'
        model = Mock(return_value={'text': output})
        self.assertEqual(attribute_rewriter('', [], kind='image', qwen=model, seed=1), output)
        first = model.call_args.args[0]
        self.assertEqual(model.call_args.args[1], [])
        self.assertIn('one or two concise sentences', first)
        attribute_rewriter('', [], kind='image', qwen=model, seed=2)
        self.assertNotEqual(first, model.call_args.args[0])
        with self.assertRaises(ValueError):
            attribute_rewriter('', [self.ref], kind='image', qwen=model)
        model.return_value={'text':'A hat. (from <Image 1>)'}
        self.assertIsNone(attribute_rewriter('', [], kind='image', qwen=model))

    def test_cross_source_repair_and_unknown_sources(self):
        other = Reference('artifact_24', self.ref.path, 'image')
        sampled = 'A blue coat. (from <Image 1>)\nA red chair. (from <Image 2>)'
        model = Mock(return_value={'text':sampled})
        self.assertEqual(attribute_rewriter(sampled, [self.ref,other], kind='image', qwen=model), sampled)
        self.assertIn('smallest necessary change', model.call_args.args[0])
        self.assertEqual(model.call_args.args[1], [self.ref,other])
        for invalid in ('A hat. (from <Image 3>)', 'A hat from <Image 1>.',
                        '{"description":"A hat."}', 'CANNOT_FULFILL'):
            model.return_value={'text': invalid}
            self.assertIsNone(attribute_rewriter(sampled, [self.ref,other], kind='image', qwen=model))

    def test_verifier_observes_only_new_media_and_receives_exact_plan(self):
        plan = 'A white hat. (from <Video 2>)\nA mirrored floor.'
        model = Mock(return_value={'text':'A black hat rests on a mirrored floor.'})
        result = attribute_verifier(generated_media=self.ref, description=plan, qwen=model)
        self.assertEqual(result, model.return_value['text'])
        self.assertIn(plan, model.call_args.args[0])
        self.assertEqual(model.call_args.args[1], [self.ref])
        self.assertEqual(model.call_count, 1)
        model.return_value={'text':'A white hat. (from <Video 2>)'}
        self.assertIsNone(attribute_verifier(generated_media=self.ref, description=plan, qwen=model))

    def test_prompts_have_roles_examples_and_grounding(self):
        for system in (extraction_system_prompt(), REWRITE, VERIFY):
            self.assertTrue(system.startswith('You are '))
            self.assertIn('example', system.lower())
            self.assertIn('ONLY', system)
        self.assertIn('fundamentally', REWRITE)
        self.assertIn('high-end', REWRITE)
        self.assertIn('grotesquerie', REWRITE)
        self.assertIn('inheritance', VERIFY)


if __name__ == '__main__': unittest.main()
