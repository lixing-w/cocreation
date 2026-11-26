import json,unittest
from pathlib import Path
from scripts.dialogue_generation.planner import Planner
from scripts.dialogue_generation.planner_rules import media_rules

ROOT=Path(__file__).resolve().parents[1]
def spec(effect,**extra):return dict(scope='global',selector={},effect=effect,text='',**extra)

class PlannerRuleTests(unittest.TestCase):
    def planner(self):
        source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        source['constraints']=[]
        return Planner(source,seed=1,reworks=0,extra_behaviors=False,cross_turn_rules=False)

    def result(self,p):return dict(placeholders=list(p.media.values()),target_assets=p.assets)

    def test_historical_snapshot_keeps_only_values_active_at_that_generation(self):
        p=self.planner();ratio=p.rules.start(spec(dict(aspect_ratio='portrait')))
        season=p.rules.start(spec(dict(season='fall')));p.single('bag','direct');before=p.ready['bag']
        p.rules.end(season);p.rules.replace([ratio],spec(dict(aspect_ratio='landscape')))
        p.single('cup','direct');after=p.ready['cup']
        old=media_rules(self.result(p),before);latest=media_rules(self.result(p),after)
        self.assertIn('9:16',str(old));self.assertIn('fall',str(old))
        self.assertIn('16:9',str(latest));self.assertNotIn('9:16',str(latest));self.assertNotIn('fall',str(latest))
        self.assertTrue(all(set(r)=={'requirement'} for r in latest))

    def test_one_request_exception_and_kind_scope(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')));p.single('bag','direct')
        with p.rules.exception({rule['id']:dict(aspect_ratio='1:1')},kind='image'):
            image=p.add_media('image','Still of the bag.',assets=['bag'],operation='generate')
            video=p.add_media('video','Clip of the bag.',assets=['bag'],operation='generate')
        followup=p.add_media('image','Another still of the bag.',assets=['bag'],operation='generate')
        result=self.result(p)
        self.assertIn('1:1',str(media_rules(result,image)))
        self.assertIn('9:16',str(media_rules(result,video)))
        self.assertIn('9:16',str(media_rules(result,followup)))

    def test_local_attributes_are_not_promoted_to_cross_turn_rules(self):
        p=self.planner();p.single('bag','direct')
        self.assertEqual(media_rules(self.result(p),p.ready['bag']),[])

    def test_assembly_rules_only_constrain_canvas_not_existing_content(self):
        result=dict(placeholders=[dict(placeholder='[[video_0004]]',kind='video',asset_ids=[],
            render_state=dict(season='winter',aspect_ratio='square_assets_landscape_video'))])
        rules=media_rules(result,'[[video_0004]]',operation='concatenate')
        self.assertEqual(rules,[dict(requirement='Use landscape 16:9.')])

    def test_child_generation_filters_subjects_and_projects_kind(self):
        result=dict(target_assets={'a':dict(name='city'),'b':dict(name='vehicle')},placeholders=[
            dict(placeholder='[[video_0001]]',kind='video',asset_ids=['a','b'],render_state=dict(
                aspect_ratio='square_assets_landscape_video',asset_attributes={'a':dict(material='fabric'),'b':dict(color='red')}))])
        rules=media_rules(result,'[[video_0001]]',kind='image',asset_ids=['a'])
        self.assertIn('1:1',str(rules));self.assertIn('fabric',str(rules));self.assertNotIn('red',str(rules))
        self.assertNotIn('vehicle',str(rules));self.assertNotIn('16:9',str(rules))

if __name__=='__main__':unittest.main()
