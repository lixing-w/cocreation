"""Media ordering contracts before the official processor encodes content."""
import unittest
from cocreation.worker import qwen_messages
from tools.find_video_frame_prompts import selection_prompt


class Interleaving(unittest.TestCase):
    def content(self, prompt, kinds):
        counts = {}
        refs = []
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
            refs.append(dict(kind=kind, label=f'<{kind.title()} {counts[kind]}>',
                             path=f'{kind}{counts[kind]}'))
        return qwen_messages(dict(prompt=prompt, references=refs,
            options=dict(max_pixels=123, video_fps=2, video_max_frames=24)))[0]['content']

    def test_repeated_and_unknown_labels_preserve_text(self):
        prompt = 'Look <Image 1>, compare <Image 1> with <Image 9>.'
        items = self.content(prompt, ['image'])
        self.assertEqual([x['type'] for x in items], ['text', 'image', 'text'])
        self.assertEqual(''.join(x['text'] for x in items if x['type']=='text'), prompt)
        self.assertEqual(items[0]['text'], 'Look <Image 1>')

    def test_unmentioned_media_precede_prompt_in_reference_order(self):
        items = self.content('Question', ['video', 'image'])
        self.assertEqual([x['type'] for x in items], ['text','video','text','image','text'])
        self.assertEqual(items[-1]['text'], 'Question')
        self.assertEqual(items[1]['fps'], 2)

    def test_empty_and_no_media(self):
        self.assertEqual(self.content('', []), [dict(type='text', text='')])
        self.assertEqual(self.content('Look <Image 1>', []),
                         [dict(type='text', text='Look <Image 1>')])

    def test_refinement_candidates_are_chronological_even_with_query_labels(self):
        prompt = selection_prompt('Same subject as <Image 3>', 3)
        items = self.content(prompt, ['image'] * 3)
        self.assertEqual([x['image'] for x in items if x['type']=='image'],
                         ['image1','image2','image3'])
        self.assertTrue(items[0]['text'].endswith('Candidates: <Image 1>'))
        self.assertEqual(sum(x['type']=='image' for x in items), 3)
