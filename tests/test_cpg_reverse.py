import copy
import io
from contextlib import redirect_stderr
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from cocreation import Reference
from tools._attribute_protocol import (atomize, assemble, sample_selection, dependencies,
    apply_variations, expert_rows, attach_provenance, validate_matches)
from tools.imaginer import imaginer
from tools.attribute_verifier import attribute_verifier
from scripts.cpg import cpg_reverse as cpg


def observation(artifact='artifact_0', kind='image'):
    return dict(artifact_id=artifact, kind=kind, attributes={
        'identity': [{'uid':'cup', 'description':'A ceramic cup.',
                      'appearance':['Red glaze.', 'A curved handle.']}],
        'environment':[{'uid':'table', 'description':'A wooden tabletop.'}],
        'identity_background_relations':[{'uid':'on', 'description':'The cup rests on the tabletop.',
                                          'links':['cup','table']}],
        'style':'Studio photography.'})


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_and_multi_parent_namespaces(self):
        one, two = atomize(observation()), atomize(observation('artifact_1'))
        self.assertTrue(set(r['id'] for r in one).isdisjoint(r['id'] for r in two))
        combined = [r for r in one+two if r['category'] != 'style']
        result = assemble(combined, 'artifact_2', 'image')
        self.assertEqual(len(result['attributes']['identity']), 2)
        self.assertEqual(result['attributes']['identity_background_relations'][1]['links'],
                         ['artifact_1_cup','artifact_1_table'])
        self.assertEqual(len(atomize(result)), len(combined))

    def test_minimum_k_dependency_closure_and_determinism(self):
        pool = atomize(observation())+atomize(observation('artifact_1'))
        deps = dependencies(pool)
        for k in range(1, 12):
            first = sample_selection(pool, k, 'image', random.Random(8))
            self.assertEqual(first, sample_selection(pool, k, 'image', random.Random(8)))
            if first is not None:
                ids = {r['id'] for r in first}
                self.assertGreaterEqual(len(ids), k)
                for uid in ids:
                    self.assertTrue(deps[uid] <= ids)
                assemble(first, 'artifact_3', 'image')
        self.assertIsNone(sample_selection(pool, 99, 'image', random.Random(0)))

    def test_dependency_overshoot_returns_immediately(self):
        pool=atomize(observation())
        relation=next(r for r in pool if r['category']=='identity_background_relations')
        def relation_first(candidates, rng):
            yield next(r for r in candidates if r['id']==relation['id'])
            self.fail('Sampler should stop after the first dependency closure exceeds K')
        with patch('tools._attribute_protocol.stratified_order', side_effect=relation_first) as order:
            selected=sample_selection(pool,2,'image',random.Random(1))
        self.assertEqual(len(selected),3)  # relation + both endpoint descriptions
        order.assert_called_once()
        self.assertEqual({r['category'] for r in selected},
                         {'identity_background_relations','identity.description','environment.description'})

    def test_category_balance_despite_large_identity_list_and_trace(self):
        from collections import Counter
        value=observation()
        value['attributes'].pop('identity_background_relations')
        value['attributes']['identity'][0]['appearance']=[f'Visible detail {i}.' for i in range(100)]
        value['attributes']['viewpoint']=[{'uid':'view','description':'Wide frontal view.'}]
        pool=atomize(value)
        counts=Counter()
        for seed in range(240):
            trace=[]
            selected=sample_selection(pool,1,'image',random.Random(seed),trace=trace)
            self.assertEqual(len(trace),1)
            counts[trace[0]['category'].split('.')[0]]+=1
            self.assertEqual(set(trace[0]['added_ids']),{r['id'] for r in selected})
            self.assertEqual(set(trace[0]['dependency_ids']),
                             set(trace[0]['added_ids'])-{trace[0]['seed_atom_id']})
        self.assertEqual(set(counts),{'identity','environment','style','viewpoint'})
        self.assertTrue(all(30 <= count <= 90 for count in counts.values()),counts)

    def test_changes_and_image_eligibility(self):
        value = observation(kind='video')
        value['attributes']['identity'][0].update(appearance_change='Red glaze turns blue.',
                                                 motion=['The cup rotates.'])
        rows = atomize(value)
        change = next(r for r in rows if r['category'].endswith('appearance_change'))
        self.assertEqual(len(dependencies(rows)[change['id']]), 3)
        selected = sample_selection(rows, 6, 'image', random.Random(2))
        self.assertFalse(any(r['category'].endswith(('motion','change')) for r in selected))
        self.assertIsNone(sample_selection(rows, len(rows), 'image', random.Random(2)))

    def test_variations_only_patch_requested_text_and_expert_sources(self):
        original = atomize(observation())
        vary = original[1]['id']
        changed = apply_variations(original, [{'id':vary,'text':'Transparent leaf-textured glass.'}], [vary])
        self.assertEqual(original[1]['text'], 'Red glaze.')
        self.assertEqual(changed[1]['owner_uid'], original[1]['owner_uid'])
        rows = expert_rows(changed)
        self.assertEqual(rows[1]['artifact_ids'], [])
        self.assertEqual(rows[0]['id'], rows[1]['parent_id'])
        for invalid in ([], [{'id':vary,'text':'Red glaze.'}], [{'id':'wrong','text':'Blue.'}],
                        [{'id':vary,'text':'Blue.','links':[]}], [{'id':vary,'text':None}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                apply_variations(original, invalid, [vary])

    def test_provenance_earliest_and_unmatched_are_new(self):
        old0, old2, new = atomize(observation()), atomize(observation('artifact_2')), atomize(observation('artifact_3'))
        matches = [{'observed_id':new[1]['id'],'source_id':old2[1]['id']},
                   {'observed_id':new[1]['id'],'source_id':old0[1]['id']}]
        prov = attach_provenance(new, matches, old0+old2)
        self.assertEqual(prov[new[1]['id']]['source_artifact_id'], 'artifact_0')
        self.assertEqual(prov[new[0]['id']]['source_artifact_id'], 'artifact_3')
        next_rows = atomize(observation('artifact_3'), prov)
        self.assertEqual(next_rows[1]['carrier_artifact_id'], 'artifact_3')
        self.assertEqual(next_rows[1]['source_artifact_id'], 'artifact_0')
        later = atomize(observation('artifact_4'))
        next_prov = attach_provenance(later,[{'observed_id':later[1]['id'],'source_id':next_rows[1]['id']}],next_rows)
        self.assertEqual(next_prov[later[1]['id']]['source_artifact_id'], 'artifact_0')

    def test_matches_reject_unknown_categories_duplicates_and_owner_conflicts(self):
        old = atomize(observation())
        new = atomize(observation('artifact_2'))
        valid = {'observed_id':new[0]['id'], 'source_id':old[0]['id']}
        for invalid in ([dict(valid, source_id='not-selected')],
                        [dict(valid, source_id=old[1]['id'])], [valid, valid]):
            with self.assertRaises(ValueError): validate_matches(invalid, new, old)
        extra = copy.deepcopy(old[1]); extra.update(id='artifact_0_other',owner_uid='artifact_0_othercup')
        with self.assertRaises(ValueError):
            validate_matches([valid, {'observed_id':new[1]['id'],'source_id':extra['id']}],new,old+[extra])


    def test_existing_expert_adapter_accepts_kept_and_new_values(self):
        from tools._prompter import prepare_inputs
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.png'; path.touch()
            original=atomize(observation())
            vary=original[1]['id']
            planned=apply_variations(original,[{'id':vary,'text':'Blue glaze.'}],[vary])
            rows=expert_rows(planned)
            checked, refs, catalog=prepare_inputs(rows,[Reference('artifact_0',path,'image')],'flux')
            self.assertEqual(len(checked),6)
            self.assertEqual(catalog[0]['generator_label'],'<Image 1>')
            self.assertEqual(checked[1]['artifact_ids'],[])

    def test_timeline_uid_references_follow_namespacing(self):
        value={'artifact_id':'artifact_0','kind':'video','attributes':{
            'environment':[{'uid':'room','description':'A small room.'},
                           {'uid':'garden','description':'A sunny garden.'}],
            'environment_change':'A cut changes room to garden.'}}
        rows=atomize(value)
        result=assemble(rows,'artifact_1','video')
        self.assertEqual(result['attributes']['environment_change'],
                         'A cut changes artifact_0_room to artifact_0_garden.')
        self.assertEqual(len(dependencies(rows)[rows[-1]['id']]),2)
        self.assertEqual(sample_selection(rows,3,'video',random.Random(1)),rows)

    def test_walkthrough_fixture_matches_actual_adapters(self):
        from scripts.cpg.walkthrough import walkthrough
        fixture=json.loads((Path(__file__).resolve().parents[1]/'scripts/cpg/examples/walkthrough.json').read_text())
        self.assertEqual(walkthrough(),fixture)
        self.assertEqual(len(fixture['next_candidate_pool']),6)
        self.assertEqual(fixture['accepted_node']['provenance']['artifact_2_a0001']['source_artifact_id'],
                         'artifact_2')

    def test_shared_atoms_choose_earliest_candidate_not_text_or_origin_media(self):
        from tools._attribute_protocol import canonicalize_shared
        early=atomize(observation('artifact_2'))
        late=atomize(observation('artifact_10'))
        for left,right in zip(early,late):
            right['source_artifact_id']=left['source_artifact_id']='artifact_0'
            right['source_attribute_uid']=left['source_attribute_uid']='origin_'+left['id']
        merged=canonicalize_shared(late+early)
        self.assertEqual(len(merged),len(early))
        self.assertEqual({r['carrier_artifact_id'] for r in merged},{'artifact_2'})
        # Equal text without verified common provenance remains two distinct atoms.
        independent=canonicalize_shared(atomize(observation())+atomize(observation('artifact_1')))
        self.assertEqual(len(independent),12)

    def test_shared_description_rebinds_later_children_and_relation_dependencies(self):
        from tools._attribute_protocol import canonicalize_shared
        early=atomize(observation())
        late=atomize(observation('artifact_1'))
        late[0]['source_artifact_id']=early[0]['source_artifact_id']
        late[0]['source_attribute_uid']=early[0]['source_attribute_uid']
        pool=early+late
        selected=sample_selection(pool,10,'image',random.Random(1),parent_count=2)
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected),10)
        target=assemble(selected,'artifact_2','image')
        self.assertEqual(len(target['attributes']['identity']),1)
        for relation in target['attributes']['identity_background_relations']:
            self.assertEqual(relation['links'][0],'artifact_0_cup')
        self.assertEqual({r['carrier_artifact_id'] for r in selected},{'artifact_0','artifact_1'})
        # Earliest description dependency counts as another actual parent.
        unique_late=[r for r in canonicalize_shared(pool) if r['carrier_artifact_id']=='artifact_1']
        self.assertTrue(any('artifact_0_a0000' in r['dependency_ids'] for r in unique_late))

    def test_shared_viewpoint_timeline_points_to_canonical_shot(self):
        one={'artifact_id':'artifact_0','kind':'video','attributes':{
            'viewpoint':[{'uid':'first','description':'A wide view.'}]}}
        two={'artifact_id':'artifact_1','kind':'video','attributes':{
            'viewpoint':[{'uid':'first','description':'A wide view.'},
                         {'uid':'second','description':'A close view.'}],
            'viewpoint_change':'A cut changes first to second.'}}
        old, later=atomize(one),atomize(two)
        later[0].update(source_artifact_id='artifact_0',source_attribute_uid=old[0]['id'])
        selected=sample_selection(old+later,3,'video',random.Random(1),parent_count=2)
        target=assemble(selected,'artifact_2','video')
        self.assertEqual(target['attributes']['viewpoint_change'],
                         'A cut changes artifact_0_first to artifact_1_second.')

    def test_parent_count_budget_applies_to_actual_carriers(self):
        pool=atomize(observation())+atomize(observation('artifact_1'))
        self.assertIsNone(sample_selection(pool,7,'image',random.Random(1),parent_count=1))
        selected=sample_selection(pool,7,'image',random.Random(1),parent_count=2)
        self.assertEqual(len({r['carrier_artifact_id'] for r in selected}),2)
        single=sample_selection(pool,6,'image',random.Random(1),parent_count=1)
        self.assertEqual(len({r['carrier_artifact_id'] for r in single}),1)
        for bad in (0,True,5.5):
            with self.assertRaises(ValueError):
                sample_selection(pool,1,'image',random.Random(1),parent_count=bad)

    def test_image_reference_hard_cap_even_if_parent_budget_is_larger(self):
        pool=[]
        for i in range(5):
            value={'artifact_id':f'artifact_{i}','kind':'image','attributes':{
                'viewpoint':[{'uid':'shot','description':'A frontal view.'}]}}
            pool.extend(atomize(value))
        self.assertIsNone(sample_selection(pool,5,'image',random.Random(1),parent_count=10))
        self.assertEqual(len(sample_selection(pool,5,'video',random.Random(1),parent_count=10)),5)



class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'media.png'; self.path.touch()
        self.old = Reference('artifact_0', self.path, 'image')
        self.new = Reference('artifact_1', self.path, 'image')
        self.rows = atomize(observation())
        self.observed = observation('artifact_1')

    def test_imaginer_exact_edits_and_no_call_for_zero(self):
        model = Mock(return_value={'text':json.dumps({'edits':[{'id':self.rows[1]['id'],'text':'Blue glaze.'}]})})
        self.assertEqual(imaginer(self.rows,[],kind='image',qwen=model),[])
        model.assert_not_called()
        result = imaginer(self.rows,[self.rows[1]['id']],[self.old],kind='image',qwen=model)
        self.assertEqual(result[0]['text'],'Blue glaze.')
        model.close.assert_not_called()
        for text in ('null','bad','{"edits":[]}','{"edits":[],"edits":[]}',
                     '{"edits":[{"id":"missing","text":"Blue."}]}'):
            model.return_value={'text':text}
            self.assertIsNone(imaginer(self.rows,[self.rows[1]['id']],[self.old],kind='image',qwen=model))

    def test_revision_preserves_contract_and_only_sees_new_media(self):
        model=Mock(return_value={'text':json.dumps(self.observed)})
        result=attribute_verifier(mode='revise_observation',generated_media=self.new,
                                 target_attributes=self.observed,qwen=model)
        self.assertEqual(result,self.observed)
        self.assertEqual(model.call_args.args[1],[self.new])
        invalid=copy.deepcopy(self.observed)
        invalid['attributes']['identity'][0]['motion']=['The cup spins.']
        model.return_value={'text':json.dumps(invalid)}
        self.assertIsNone(attribute_verifier(mode='revise_observation',generated_media=self.new,
                                             target_attributes=self.observed,qwen=model))

    def test_verifier_revision_uses_same_conservative_repairs(self):
        value=copy.deepcopy(self.observed)
        value['attributes']['style']='unknown'
        value['attributes']['identity'][0]['appearance']=['Violet glaze.',None]
        model=Mock(return_value={'text':json.dumps(value)})
        result=attribute_verifier(mode='revise_observation',generated_media=self.new,
                                 target_attributes=self.observed,qwen=model)
        self.assertNotIn('style',result['attributes'])
        self.assertEqual(result['attributes']['identity'][0]['appearance'],['Violet glaze.'])
        self.assertIn('style',self.observed['attributes'])

    def test_verifier_discards_draft_and_imaginer_has_no_word_limit(self):
        from tools.attribute_observation_prompts import DRAFT_KEY
        value=copy.deepcopy(self.observed)
        value['attributes']={DRAFT_KEY:{'identity':[], 'environment':[]}, **value['attributes']}
        model=Mock(return_value={'text':json.dumps(value)})
        result=attribute_verifier(mode='revise_observation',generated_media=self.new,
                                 target_attributes=self.observed,qwen=model)
        self.assertEqual(result,self.observed)
        text='A deliberately lengthy desired description ' + 'detail '*25
        result=apply_variations(self.rows,[{'id':self.rows[1]['id'],'text':text}], [self.rows[1]['id']])
        self.assertEqual(result[1]['text'],text)

    def test_second_pass_only_selected_originals_and_empty_vs_null(self):
        selected=self.rows[:2]
        sources=[dict(reference=self.old,attributes=selected)]
        model=Mock(return_value={'text':'{"matches":[]}'})
        result=attribute_verifier(mode='verify_inheritance',generated_media=self.new,
            observed_attributes=self.observed,selected_sources=sources,qwen=model)
        self.assertEqual(result,[])
        payload=json.loads(model.call_args.args[0].split('Task data:\n')[1])
        self.assertEqual(len(payload['selected_sources'][0]['attributes']),2)
        self.assertEqual(payload['selected_sources'][0]['attributes'][1]['text'],'Red glaze.')
        self.assertEqual(model.call_args.args[1],[self.new,self.old])
        newrows=atomize(self.observed)
        model.return_value={'text':json.dumps({'matches':[{'observed_id':newrows[2]['id'],
                                                          'source_id':self.rows[2]['id']}]})}
        self.assertIsNone(attribute_verifier(mode='verify_inheritance',generated_media=self.new,
            observed_attributes=self.observed,selected_sources=sources,qwen=model))
        model.return_value={'text':'null'}
        self.assertIsNone(attribute_verifier(mode='verify_inheritance',generated_media=self.new,
            observed_attributes=self.observed,selected_sources=sources,qwen=model))

    def test_owned_model_closes_on_error_and_shared_does_not(self):
        model=Mock(side_effect=RuntimeError('worker failed'))
        with patch('tools._attribute_vlm.Qwen',return_value=model):
            with self.assertRaises(RuntimeError):
                imaginer(self.rows,[self.rows[1]['id']],[self.old],kind='image')
        model.close.assert_called_once()
        model.reset_mock()
        with self.assertRaises(RuntimeError):
            imaginer(self.rows,[self.rows[1]['id']],[self.old],kind='image',qwen=model)
        model.close.assert_not_called()


class FakeModel:
    def __init__(self, path, kind):
        self.path, self.kind, self.calls, self.closed = path, kind, [], 0
    def __enter__(self): return self
    def __exit__(self,*args): self.closed += 1
    def __call__(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return {'path':str(self.path),'kind':self.kind,'status':'ok'}


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.media=self.base/'source.png'; self.media.touch()
        self.manifest=self.base/'roots.json'
        self.manifest.write_text(json.dumps([dict(artifact_id='artifact_0',kind='image',
            path='source.png',observation=observation())]))
        self.qwen=FakeModel(self.media,'image'); self.flux=FakeModel(self.media,'image')
        self.vdn=FakeModel(self.media,'video')
        for name,value in [('Qwen',self.qwen),('Flux',self.flux),('VDNH3',self.vdn)]:
            mock=self.enterContext(patch.object(cpg,name,return_value=value))
            setattr(self,'factory_'+name,mock)
        self.extract=self.enterContext(patch.object(cpg,'extract_attribute_nodes',return_value=observation()))
        self.imagine=self.enterContext(patch.object(cpg,'imaginer',return_value=[]))
        self.flux_expert=self.enterContext(patch.object(cpg,'flux_prompter',return_value='Prompt.'))
        self.video_expert=self.enterContext(patch.object(cpg,'h3_prompter',return_value='Prompt.'))
        def verify(**kwargs):
            if kwargs['mode']=='revise_observation': return kwargs['target_attributes']
            return []
        self.verify=self.enterContext(patch.object(cpg,'attribute_verifier',side_effect=verify))
        self.frame=self.enterContext(patch.object(cpg,'find_video_frame',return_value=None))

    def run_loop(self,**kwargs):
        options=dict(attempts=2,keep=(6,6),variate=(0,0),p_video=0,seed=0)
        options.update(kwargs)
        return cpg.run(self.manifest,self.base/'run',**options)

    def test_stage_debug_console_and_persistent_log(self):
        stream=io.StringIO()
        with redirect_stderr(stream):
            graph=self.run_loop(attempts=1)
        output=stream.getvalue()
        for label in ('extractor or supplied observation result','parent atom pool',
                      'candidate pool','actual selection','imaginer result',
                      'generation expert result','generation result',
                      'verifier pass 1 result','verifier pass 2 result','final provenance','outcome'):
            self.assertIn(label,output)
        self.assertIn('artifact_0',output)
        self.assertIn('Red glaze.',output)
        self.assertEqual((self.base/'run/debug.log').read_text(),output)
        self.assertEqual(graph['attempts'][0]['status'],'accepted')

    def test_prompter_diagnostic_copied_to_attempt_and_debug_log(self):
        def decline(*args,**kwargs):
            path=Path(kwargs['output_dir']);path.mkdir(parents=True)
            (path/'diagnostics.jsonl').write_text(json.dumps({'code':'h3_sections','message':'Missing heading'})+'\n')
            return None
        self.flux_expert.side_effect=decline
        graph=self.run_loop(attempts=1,debug=False)
        self.assertEqual(graph['attempts'][0]['prompter_error']['code'],'h3_sections')
        self.assertIn('Missing heading',(self.base/'run/debug.log').read_text())

    def test_quiet_keeps_debug_file_without_console_dumps(self):
        stream=io.StringIO()
        with redirect_stderr(stream): self.run_loop(attempts=1,debug=False)
        self.assertEqual(stream.getvalue(),'')
        self.assertIn('verifier pass 2 result',(self.base/'run/debug.log').read_text())

    def test_models_once_root_only_extraction_and_new_nodes_reused(self):
        graph=self.run_loop(attempts=6)
        self.assertEqual(len(graph['artifacts']),7)
        for factory in (self.factory_Qwen,self.factory_Flux,self.factory_VDNH3): factory.assert_called_once()
        self.assertEqual([self.qwen.closed,self.flux.closed,self.vdn.closed],[1,1,1])
        self.extract.assert_not_called()
        self.assertTrue(any(any(p!='artifact_0' for p in a['sampled_parents']) for a in graph['attempts']))
        self.assertTrue(all(call.kwargs['qwen'] is self.qwen for call in self.verify.call_args_list))
        self.assertEqual(json.loads((self.base/'run/graph.json').read_text()),graph)

    def test_each_null_stage_skips_and_next_attempt_runs(self):
        # Each subcase uses a fresh output folder and no retry within the trial.
        for stage in ('imaginer','expert','revision','inheritance','generation'):
            with self.subTest(stage=stage):
                self.imagine.side_effect=None; self.imagine.return_value=[]
                self.flux_expert.side_effect=None; self.flux_expert.return_value='Prompt.'
                state={'failed':False}
                def verify(**kw):
                    target='revise_observation' if stage=='revision' else 'verify_inheritance'
                    if stage in ('revision','inheritance') and kw['mode']==target and not state['failed']:
                        state['failed']=True; return None
                    return kw['target_attributes'] if kw['mode']=='revise_observation' else []
                self.verify.side_effect=verify
                if stage=='imaginer': self.imagine.side_effect=[None,[]]
                if stage=='expert': self.flux_expert.side_effect=[None,'Prompt.']
                original=self.flux.__class__.__call__
                def generate(instance,*args,**kw):
                    if stage=='generation' and not state['failed']:
                        state['failed']=True; return None
                    return original(instance,*args,**kw)
                with patch.object(FakeModel,'__call__',generate):
                    graph=cpg.run(self.manifest,self.base/stage,attempts=2,keep=(6,6),
                                  variate=(0,0),p_video=0)
                self.assertEqual([a['status'] for a in graph['attempts']],['skipped','accepted'])
                self.assertEqual(len(graph['artifacts']),2)

    def test_video_branch_and_original_text_to_second_pass(self):
        self.imagine.side_effect=lambda selected,ids,*a,**kw: [{'id':uid,'text':'A surreal visual reinterpretation.'} for uid in ids]
        graph=self.run_loop(attempts=1,p_video=1,variate=(1,1))
        self.assertEqual(graph['attempts'][0]['status'],'accepted')
        self.assertEqual(len(self.vdn.calls),1)
        self.assertEqual(len(self.flux.calls),0)
        source=self.verify.call_args_list[1].kwargs['selected_sources'][0]['attributes']
        self.assertEqual([r['text'] for r in source],[r['text'] for r in atomize(observation())])
        self.assertEqual(self.video_expert.call_args.kwargs['seconds'],5)

    def test_frame_null_skips_without_generation(self):
        value=observation(kind='video')
        self.manifest.write_text(json.dumps([dict(artifact_id='artifact_0',kind='video',path='source.png',observation=value)]))
        graph=self.run_loop(attempts=1)
        self.assertEqual(graph['attempts'][0]['stage'],'frame_search')
        self.assertEqual(self.flux.calls,[])

    def test_missing_root_observation_extracted_once(self):
        self.manifest.write_text(json.dumps([dict(artifact_id='artifact_0',kind='image',path='source.png')]))
        self.run_loop()
        self.extract.assert_called_once()
        self.assertIs(self.extract.call_args.kwargs['qwen'],self.qwen)

    def test_impossible_counts_skip_and_no_tools(self):
        graph=self.run_loop(keep=(100,100))
        self.assertTrue(all(a['status']=='skipped' for a in graph['attempts']))
        self.imagine.assert_not_called()
        self.verify.assert_not_called()

    def test_worker_failure_checkpoints_and_closes(self):
        self.flux_expert.side_effect=RuntimeError('worker failed')
        with self.assertRaises(RuntimeError): self.run_loop()
        graph=json.loads((self.base/'run/graph.json').read_text())
        self.assertEqual(graph['attempts'][0]['status'],'error')
        self.assertEqual([self.qwen.closed,self.flux.closed,self.vdn.closed],[1,1,1])


    def test_two_parent_loop_and_positive_frame_transport(self):
        self.manifest.write_text(json.dumps([
            dict(artifact_id='artifact_0',kind='image',path='source.png',observation=observation()),
            dict(artifact_id='artifact_1',kind='video',path='source.png',observation=observation('artifact_1','video'))]))
        def frame(*args,**kwargs):
            Path(kwargs['frame_output']).touch()
            return 1.5
        self.frame.side_effect=frame
        graph=self.run_loop(attempts=8,keep=(9,9))
        accepted=[a for a in graph['attempts'] if a['status']=='accepted']
        self.assertTrue(accepted)
        self.assertEqual(len(graph['artifacts'][2]['parents']),2)
        self.assertTrue(any(a.get('frames') for a in accepted))
        for args, kwargs in self.flux.calls:
            self.assertTrue(all(r.kind=='image' for r in args[1]))
        self.assertTrue(any(r.kind=='video' for call in self.flux_expert.call_args_list
                            for r in call.args[1]))
        self.assertTrue(any(s['reference'].kind=='video' for call in self.verify.call_args_list
                            if call.kwargs['mode']=='verify_inheritance' for s in call.kwargs['selected_sources']))

    def test_null_root_leaves_no_usable_parents(self):
        self.manifest.write_text(json.dumps([dict(artifact_id='artifact_0',kind='image',path='source.png')]))
        self.extract.return_value=None
        graph=self.run_loop()
        self.assertEqual(graph['artifacts'],[])
        self.assertEqual(graph['root_failures'],['artifact_0'])
        self.assertTrue(all(a['reason']=='no usable parents' for a in graph['attempts']))
        self.imagine.assert_not_called()

if __name__=='__main__': unittest.main()
