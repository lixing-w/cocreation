"""Deterministic fictional JSON walkthrough; no models or media generation."""
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._attribute_protocol import atomize, assemble, apply_variations, attach_provenance, expert_rows


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
    edits = [{'id':'artifact_0_a0001','text':'Violet glaze flecked with tiny golden stars.'}]
    planned = apply_variations(selected, edits, ['artifact_0_a0001'])
    target = assemble(planned, 'artifact_2','image')
    # Pretend the generator omitted the gold stars and added legible lettering.
    observed = json.loads(json.dumps(target))
    observed['attributes']['identity'][0]['appearance'] = ['Violet glaze.']
    observed['attributes']['visible_text'] = [{'uid':'tea_label','description':'“TEA” appears in white letters on the cup.'}]
    new = atomize(observed)
    matches = [{'observed_id':'artifact_2_a0000','source_id':'artifact_0_a0000'},
               {'observed_id':'artifact_2_a0002','source_id':'artifact_1_a0000'},
               {'observed_id':'artifact_2_a0003','source_id':'artifact_1_a0002'},
               {'observed_id':'artifact_2_a0004','source_id':'artifact_1_a0003'}]
    provenance = attach_provenance(new, matches, selected)
    return dict(fictional=True, roots=roots, candidate_pool=pool,
        selection=dict(N=len(pool), K=5, V=1, dropped_ids=[r['id'] for r in pool if r['id'] not in ids],
                       selected_originals=selected),
        imaginer_output={'edits':edits}, target=target, expert_input=expert_rows(planned),
        verifier_pass_1_output=observed,
        verifier_pass_2_selected_originals=selected,
        verifier_pass_2_output={'matches':matches},
        accepted_node=dict(observation=observed, provenance=provenance, parents=['artifact_0','artifact_1']),
        next_candidate_pool=atomize(observed, provenance))


if __name__ == '__main__':
    print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
