"""Deterministic fictional JSON walkthrough; no models or media generation."""
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._attribute_protocol import atomize, assemble, apply_additions, attach_provenance, expert_rows


def walkthrough():
    roots = [
        {'artifact_id':'artifact_0','kind':'image','attributes':{
            'identity':[{'uid':'cup','description':'A ceramic cup.',
                         'appearance':['Red glaze.','A curved handle.']}],
            'environment':[{'uid':'table','description':'A wooden tabletop.'}],
            'identity_background_relations':[{'uid':'on','description':'The cup rests on the tabletop.',
                                              'links':['cup','table']}],
            'style':'Studio photography.'}},
        {'artifact_id':'artifact_1','kind':'image','attributes':{
            'environment':[{'uid':'garden','description':'A garden filled with white flowers.',
                            'location':'A walled courtyard.','lighting':'Soft morning sunlight.'}],
            'style':'Watercolor with soft pigment edges.'}}]
    pool = [r for root in roots for r in atomize(root)]
    ids = ['artifact_0_a0000','artifact_0_a0001','artifact_1_a0000','artifact_1_a0002','artifact_1_a0003']
    selected = [r for r in pool if r['id'] in ids]
    additions = [{'category':'identity','text':'A tiny glass dragon beside the cup.'}]
    planned = apply_additions(selected, additions, 1, 'image')
    target = assemble(planned, 'artifact_2','image')
    # Pretend all requested properties appeared, plus unrequested lettering.
    observed = json.loads(json.dumps(target))
    observed['attributes']['visible_text'] = [{'uid':'tea_label','description':'“TEA” appears on the cup.'}]
    new = atomize(observed)
    # Fictional oracle only; production uses verifier pass 2 with media.
    matches = [{'observed_id':row['id'],'source_id':source['id']}
               for row in new for source in selected
               if row['category']==source['category'] and row['text']==source['text']]
    provenance = attach_provenance(new, matches, selected)
    return dict(fictional=True, roots=roots, candidate_pool=pool,
        selection=dict(N=len(pool), K=5, V=1, dropped_ids=[r['id'] for r in pool if r['id'] not in ids],
                       selected_originals=selected),
        imaginer_output={'additions':additions}, target=target, expert_input=expert_rows(planned),
        verifier_pass_1_output=observed,
        verifier_pass_2_selected_originals=selected,
        verifier_pass_2_output={'matches':matches},
        accepted_node=dict(observation=observed, provenance=provenance, parents=['artifact_0','artifact_1']),
        next_candidate_pool=atomize(observed, provenance))


if __name__ == '__main__':
    print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
