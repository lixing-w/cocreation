"""Project logical design metadata into the current rendered output."""
from copy import deepcopy
import re

COLOR_FIELDS = {'color', 'hair_color', 'clothing_color', 'palette', 'water_color', 'vegetation_color',
                'color_saturation', 'color_lightness'}
CHROMATIC = (r'reddish|bluish|greenish|yellowish|pinkish|brownish|orange|purple|violet|magenta|cyan|'
             r'turquoise|teal|beige|khaki|maroon|burgundy|lavender|scarlet|crimson|golden|gold|silver|'
             r'bronze|copper|cream|tan|brown|pink|blue|green|yellow|red|multicolou?red|colou?rful|'
             r'colou?red|warm[- ]toned|cool[- ]toned')
COLOR_WORD = re.compile(r'(?i)\b(?:'+CHROMATIC+r')\b')
SATURATION = re.compile(r'(?i)\b(?:(?:very )?(?:low|medium|high) saturation|saturated|desaturated|pink hue)\b')

def monochrome(state):
    return state.get('color_mode') == 'black_and_white'

def neutral_text(text):
    text = SATURATION.sub('', COLOR_WORD.sub('', text))
    return re.sub(r' +',' ',text).replace(' ,',',').strip()

def output_contract(value, state):
    """Never give the writer contradictory hue instructions in a monochrome task."""
    if not monochrome(state): return deepcopy(value)
    if isinstance(value, str): return neutral_text(value)
    if isinstance(value, list): return [output_contract(v,state) for v in value]
    if isinstance(value, dict):
        return {k:output_contract(v,state) for k,v in value.items()
                if (k not in COLOR_FIELDS or (k.endswith('color') and v in ('white','black','gray','grey','light gray','dark gray','black and white'))) and k not in ('output_color_mode',)}
    return value

def prompt_conflicts(prompt, state):
    if not monochrome(state): return []
    errors=[]
    if COLOR_WORD.search(prompt) or SATURATION.search(prompt):
        errors.append('The entire output must be black and white. Remove all hue, colored-light and saturation instructions, including identity names that contain colors.')
    affirmative = re.sub(r'(?i)\b(?:no|without|never|do not use) selective[- ]colou?r(?: or saturation)?', '', prompt)
    if re.search(r'(?i)selective[- ]colou?r|(?:black[ -]and[ -]white|gr[ae]yscale|monochrom\w*)[^.\n]{0,100}\bexcept\b|retain[^.\n]*saturation|keep[^.\n]*(?:original|natural|full)[ -]colou?r|original colou?rs|full[- ]colou?r',affirmative):
        errors.append('No selective-color exception is authorized: every subject, foreground and background must be grayscale.')
    if not re.search(r'(?i)black[ -]and[ -]white|gr[ae]yscale|monochrom',prompt):
        errors.append('Explicitly request black-and-white rendering of the entire image or video.')
    return errors
