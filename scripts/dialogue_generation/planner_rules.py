"""Minimal current rule requirements from a media's planner snapshot; no model calls."""
from scripts.dialogue_generation.cross_turn_rule import visible_effect, effect_text, matches, scope_text
from scripts.dialogue_generation.output_geometry import aspect_ratio_rule


def media_rules(result, token, *, kind=None, asset_ids=None, operation=None):
    rows={row['placeholder']:row for row in result.get('placeholders',[])}
    if token not in rows:
        if 'placeholders' in result:
            raise ValueError('No planner rule snapshot for '+token)
        return []  # Text-only callers with no media contract have no rule snapshot.
    row=dict(rows[token])
    if kind is not None:row['kind']=kind
    if asset_ids is not None:row['asset_ids']=asset_ids
    state=visible_effect(row.get('render_state',{}),row)
    if operation in ('concatenate','copy'):
        # Existing clips retain their content; only output-canvas requirements
        # apply to a deterministic assembly/export.
        state={key:value for key,value in state.items() if key=='aspect_ratio'}
    names={aid:asset['name'] for aid,asset in result.get('target_assets',{}).items()}
    rules=[]
    for field,value in state.items():
        if field=='asset_attributes':
            for aid,attributes in value.items():
                for key,attribute in attributes.items():
                    effect={'asset_attributes':{aid:{key:attribute}}}
                    rules.append(dict(requirement='Use '+effect_text(effect,names)+'.'))
        else:
            rules.append(dict(requirement='Use '+effect_text({field:value},names)+'.'))
    return rules


def checked_rules(rules):
    if not isinstance(rules,list) or any(not isinstance(rule,dict) or set(rule)!={'requirement'}
        or not isinstance(rule['requirement'],str) or not rule['requirement'].strip() for rule in rules):
        raise ValueError('Expected current planner rules as requirement-only entries')
    return [dict(rule) for rule in rules]


def rule_applicability(result, token, *, kind=None, asset_ids=None, operation=None):
    """Explain existing planner rule scope at this output, without changing it."""
    original=next((row for row in result.get('placeholders',[]) if row['placeholder']==token),None)
    if original is None:
        return dict(effective=[],inactive=[])
    row=dict(original)
    if kind is not None:row['kind']=kind
    if asset_ids is not None:row['asset_ids']=asset_ids
    at=row.get('introduced_at')
    if at is None:
        at=next((turn.get('id',i+1) for i,turn in enumerate(result.get('turns',[]))
                 if token in turn.get('output_media',[])),None)
    names={aid:asset['name'] for aid,asset in result.get('target_assets',{}).items()}
    applied={entry['id']:entry['effect'] for entry in row.get('applied_rules',[])}
    view=dict(output_kind=row['kind'],output_subjects=[names.get(aid,aid) for aid in row.get('asset_ids',[])],
              aspect_ratio=aspect_ratio_rule(row.get('render_state', {}), row['kind']),
              effective_requirements=media_rules(result,token,kind=kind,asset_ids=asset_ids,operation=operation),
              effective=[],inactive=[])
    for rule in result.get('rules',[]):
        start=rule.get('started_at')
        if start is None or at is None or start>at:
            continue  # Future or unconfirmed rules were not yet part of the conversation.
        effect=visible_effect(rule['effect'],row)
        entry=dict(id=rule['id'],requirement='Use '+effect_text(effect or rule['effect'],names)+'.',
                   scope=scope_text(rule,names),started_at=start)
        end=rule.get('ended_at')
        if end is not None and end<=at:
            reason='This earlier rule ended before this output; its cancellation or replacement in the conversation takes precedence.'
        elif not matches(rule,row):
            reason='Its scope is '+entry['scope']+', which excludes this '+row['kind']+' output or its subjects.'
        elif not effect:
            reason='Its named subject attributes do not concern any subject in this output.'
        elif operation in ('concatenate','copy') and not set(effect)<= {'aspect_ratio'}:
            if 'aspect_ratio' in effect:
                canvas=dict(entry,requirement='Use '+effect_text({'aspect_ratio':effect['aspect_ratio']},names)+'.',
                            reason='The output-canvas part still applies to this assembly.')
                view['effective'].append(canvas)
                entry['requirement']='Use '+effect_text({k:v for k,v in effect.items() if k!='aspect_ratio'},names)+'.'
            reason='This operation assembles existing clips without regenerating their content; only output-canvas rules apply.'
        elif rule['id'] in applied and visible_effect(applied[rule['id']],row)!=effect:
            reason='The confirmed exception for this request replaces this rule treatment; use the effective requirements for this output.'
        else:
            entry['reason']='It was in force at this turn and its scope includes this output.'
            view['effective'].append(entry)
            continue
        entry['reason']=reason
        view['inactive'].append(entry)
    return view
