import unittest
from tools.dialogue_realizer import current_source_design

class CurrentSourceTests(unittest.TestCase):
    def test_obsolete_adult_description_does_not_prime_child_request(self):
        text=current_source_design('Request person. adult female with long slightly curled hair and a soft knit cardigan; apparent age group: child',
            [dict(subject='person',attributes=dict(age_group='child'))])
        self.assertNotIn('adult',text)
        self.assertNotIn('apparent age group',text)
        self.assertIn('female',text)
        self.assertIn('long slightly curled hair',text)
        self.assertIn('soft knit cardigan',text)

    def test_object_condition_and_multiple_people_ages_are_not_scrubbed(self):
        text='An adult female carrying an old leather bag.'
        self.assertIn('old leather bag',current_source_design(text,[dict(attributes=dict(age_group='child'))]))
        self.assertEqual(current_source_design(text,[dict(attributes=dict(age_group='adult')),dict(attributes=dict(age_group='child'))]),text)

if __name__=='__main__':unittest.main()
