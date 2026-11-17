# Target video observation prompt

Observe the supplied final video. Return compact JSON only. Record observations and choose enum values. Leave editing and dialogue planning to later stages.
Output video, attribute_schema_version (2), assets, shots, and target_order. Describe one to four shots.
For each asset include id, kind, name, description, frames (one or two timestamps in seconds where it is visible), and attributes.
Choose kind from person, object, indoor_environment, outdoor_environment. Track people and objects separately.
Reuse an id for the same person, object, or environment across shots. If the final video has distinct visible versions, use a separate id for each version.
Track only human-intuitive salient identities. Tiny incidental objects and decorative details belong in the owning asset description, not standalone assets. Describe distinctive details concisely. Additional supported attribute fields or values are legal per-identity extensions; prefer the listed choices for existing fields. Track the principal people, objects, and environments. List each shot's id, start, end, description (action and composition), and assets (the relevant asset ids).
List ids in final playback order in target_order. Every asset must appear in at least one shot.
For each field of each asset, select exactly one of the listed values.
Use unknown when the image does not support a judgment. Use not applicable when the field does not make sense for the asset.
Gender presentation and apparent age describe visible appearance only. Do not infer identity or actual age.
Select object size and apparent height only when a reliable visual scale is available. Otherwise use unknown.
Use description for stable details outside the catalog. Leave changeable attributes such as color, material, hairstyle, and weather to their fields.
Select the dominant value when several are visible. Use unknown when no value clearly dominates.
Write names, descriptions, and any constraint text in English. Use exact enum strings as printed below. Use short, version-neutral names, such as traveler or tote bag.
Use unknown for details such as actual material, apparent height, wind strength, and season when the video cannot establish them. Use numbers only in numeric fields.
Include constraints only when the input explicitly supplies lasting production requirements. A visible property alone does not establish a persistent user rule.
For explicit user rules only, use constraints with scope global or partial, a selector for partial scope, and a structured effect. Global means every generated image and video. Partial selectors may specify roles or asset_ids. Effects may specify color_mode (black_and_white or original_colors), season (spring, summer, fall, winter), or asset_attributes. Do not infer user rules from video appearance.
Follow the compatibility rules below.

## Attribute values

### Person (person)

| Field | Meaning | Allowed values |
| --- | --- | --- |
| gender_presentation | Gender presentation | feminine, masculine, androgynous, unknown, not applicable |
| age_group | Apparent age group | young child, school-age child, teenager, young adult, middle-aged adult, older adult, unknown, not applicable |
| hair_length | Hair length | bald, buzz-length hair, short hair, shoulder-length hair, long hair, waist-length hair, unknown, not applicable |
| hair_style | Hairstyle | sleek straight hair, voluminous straight hair, slightly wavy hair, wavy hair, tight curls, side part, center part, blunt bangs, wispy bangs, choppy bangs, slicked back, tousled hair, unknown, not applicable |
| hair_color | Hair color | black, white, gray, beige, brown, red, orange, yellow, green, cyan, blue, purple, pink, gold, silver, copper, multicolored, gray-white, silver-white, two-tone hair, highlights, dark blue, light blue, dark green, light green, burgundy, khaki, off-white, neon green, unknown, not applicable |
| face_shape | Face shape | round face, oval face, square face, long face, heart-shaped face, diamond-shaped face, unknown, not applicable |
| eye_shape | Eye shape | round eyes, almond-shaped eyes, narrow eyes, upturned eyes, downturned eyes, deep-set eyes, unknown, not applicable |
| eyebrow_shape | Eyebrow shape | straight brows, arched brows, upturned brows, thick brows, thin brows, unknown, not applicable |
| nose_shape | Nose shape | small nose, straight nose, rounded nose, wide nose, aquiline nose, unknown, not applicable |
| lip_shape | Lip shape | thin lips, medium lips, full lips, bow-shaped lips, rounded lips, unknown, not applicable |
| expression | Facial expression | serious, calm, gentle, smiling, laughing, thoughtful, surprised, sad, confident, tired, unknown, not applicable |
| makeup_intensity | Makeup intensity | no makeup, light makeup, moderate makeup, heavy makeup, stage makeup, unknown, not applicable |
| facial_hair | Facial hair | clean-shaven, stubble, mustache, goatee, full beard, long beard, unknown, not applicable |
| height | Apparent height | short, average height, tall, very tall, unknown, not applicable |
| body_build | Build | slim, proportionate, athletic, stocky, full-figured, unknown, not applicable |
| clothing_type | Main garment | coat, jacket, trench coat, down jacket, suit, button-up shirt, T-shirt, knit sweater, sweatshirt, sportswear, dress, workwear, traditional clothing, unknown, not applicable |
| clothing_color | Main clothing color | black, white, gray, beige, brown, red, orange, yellow, green, cyan, blue, purple, pink, gold, silver, copper, multicolored, dark blue, light blue, dark green, light green, burgundy, khaki, off-white, neon green, unknown, not applicable |
| clothing_fit | Clothing fit | fitted, regular fit, loose, oversized, unknown, not applicable |
| accessory | Main accessory | no accessory, eyeglasses, sunglasses, scarf, necklace, earrings, hat, hair accessory, unknown, not applicable |

If hair_length is bald, then hair_style must be unknown or not applicable, and hair_color must be unknown or not applicable.

### Object (object)

| Field | Meaning | Allowed values |
| --- | --- | --- |
| color | Main object color | black, white, gray, beige, brown, red, orange, yellow, green, cyan, blue, purple, pink, gold, silver, copper, colorless, multicolored, dark blue, light blue, dark green, light green, burgundy, khaki, off-white, neon green, unknown, not applicable |
| color_lightness | Object color lightness | very dark, somewhat dark, medium lightness, somewhat light, very light, unknown, not applicable |
| color_saturation | Object color saturation | low saturation, medium saturation, high saturation, unknown, not applicable |
| size | Object size | miniature, small, medium-sized, large, extra large, unknown, not applicable |
| shape | Object shape | round, oval, triangular, square, rectangular, rounded rectangle, cylindrical, conical, spherical, hemispherical, elongated, streamlined, spindle-shaped, flat, irregular, unknown, not applicable |
| aspect_ratio | Object proportions | long and narrow, balanced, wide and flat, unknown, not applicable |
| material | Main material | canvas, cotton fabric, leather, faux leather, nylon, silk, felt, wood, bamboo, ceramic, glass, metal, plastic, rubber, stone, paper, unknown, not applicable |
| surface_finish | Surface finish | matte, semi-matte, glossy, mirror finish, frosted, brushed, glazed, satin finish, unknown, not applicable |
| texture | Surface texture | fine-textured, smooth, grainy, rough, woven, fuzzy, carved, unknown, not applicable |
| pattern | Pattern | plain, striped, checked, polka dots, geometric pattern, floral pattern, animal print, gradient, lettering, logo, unknown, not applicable |
| decoration | Decoration | undecorated, decals, relief, embroidery, metal accents, charms, inlay, unknown, not applicable |
| condition | Condition | brand new, lightly worn, visibly worn, distressed, damaged, visibly repaired, unknown, not applicable |
| transparency | Transparency | opaque, translucent, transparent, unknown, not applicable |

If transparency is transparent or translucent, then material must be glass or plastic or unknown.
If surface_finish is brushed, then material must be metal or unknown.
If surface_finish is glazed, then material must be ceramic or unknown.
If color is black or white or gray or colorless, then color_saturation must be unknown or not applicable.
If material is canvas or cotton fabric or nylon or felt or paper, then surface_finish must be matte or semi-matte or unknown or not applicable.
If color is beige or brown or red or orange or yellow or green or cyan or blue or purple or pink or gold or silver or copper or multicolored or dark blue or light blue or dark green or light green or burgundy or khaki or off-white or neon green, then color_saturation must be low saturation or medium saturation or high saturation or unknown.
If color is colorless, then transparency must be transparent or unknown.

### Indoor environment (indoor_environment)

| Field | Meaning | Allowed values |
| --- | --- | --- |
| room_type | Room type | coffee shop, restaurant, living room, bedroom, kitchen, bathroom, study, office, studio, gallery, shop, hotel lobby, hallway, vehicle interior, warehouse, classroom, stage, unknown, not applicable |
| interior_style | Interior style | modern minimalist, Scandinavian, Japanese, Chinese, industrial, retro, country, understated luxury, bohemian, futuristic, classical, Mediterranean, unknown, not applicable |
| space_size | Room size | cramped, compact, mid-sized, spacious, vast, unknown, not applicable |
| ceiling_height | Ceiling height | low ceiling, standard ceiling, double-height ceiling, very high ceiling, unknown, not applicable |
| layout | Layout | open-plan, semi-open, partitioned, long and narrow, symmetrical, circular, unknown, not applicable |
| wall_material | Wall material | latex paint, exposed brick, wood paneling, concrete, stone, glass, tile, wallpaper, metal, unknown, not applicable |
| floor_material | Floor material | wood flooring, tile, concrete, marble, carpet, stone slabs, metal, unknown, not applicable |
| palette | Overall palette | warm palette, cool palette, neutral palette, black and white, saturated colors, muted colors, natural wood tones, unknown, not applicable |
| furniture_density | Furniture density | sparsely furnished, moderately furnished, densely furnished, unknown, not applicable |
| furniture_style | Furniture style | minimalist, natural wood, modern metal, retro, classical, eclectic, unknown, not applicable |
| window_type | Main window type | no windows, small windows, standard windows, floor-to-ceiling windows, skylight, panoramic glazing, unknown, not applicable |
| clutter | Clutter | tidy, slightly cluttered, cluttered, unknown, not applicable |
| background_crowd | Background crowd | no people, sparse crowd, moderate crowd, crowded, unknown, not applicable |
| lighting_source | Main light source | natural daylight, moonlight, light through windows, indoor lights, streetlights, neon lights, campfire, mixed light sources, unknown, not applicable |
| lighting_level | Ambient light level | dim, somewhat dim, moderate light, bright, very bright, unknown, not applicable |
| lighting_temperature | Lighting color temperature | cool-toned, neutral-toned, warm-toned, colored lighting, unknown, not applicable |
| lighting_softness | Light quality | hard light, moderately soft light, soft light, diffuse light, unknown, not applicable |
| time_of_day | Time of day | dawn, early morning, morning, noon, afternoon, dusk, night, late night, unknown, not applicable |
| camera_distance | Shot size | long shot, wide shot, medium shot, close shot, unknown, not applicable |

If time_of_day is night or late night, then lighting_source must be moonlight or light through windows or indoor lights or streetlights or neon lights or campfire or mixed light sources or unknown or not applicable.

### Outdoor environment (outdoor_environment)

| Field | Meaning | Allowed values |
| --- | --- | --- |
| setting_type | Setting type | city street, alley, public square, park, garden, forest, grassland, mountains, lake, river, beach, open sea, desert, snowfield, countryside, field, campsite, unknown, not applicable |
| geography | Terrain | plain, hills, valley, alpine terrain, basin, coast, island, canyon, unknown, not applicable |
| season | Season | spring, summer, fall, winter, unknown, not applicable |
| weather | Weather | clear skies, partly cloudy, overcast, light rain, heavy rain, thunderstorm, light snow, heavy snow, foggy, blowing dust, unknown, not applicable |
| surface_material | Main ground surface | asphalt, stone slabs, gravel, sand, grass, wooden boardwalk, ice and snow, dirt, rock, water surface, unknown, not applicable |
| surface_condition | Ground condition | dry, wet, standing water, snow-covered, icy, muddy, unknown, not applicable |
| vegetation_density | Vegetation density | no vegetation, sparse vegetation, moderate vegetation, dense vegetation, unknown, not applicable |
| vegetation_color | Main vegetation color | bright green, deep green, yellow-green, golden yellow, orange-red, brown, snow white, unknown, not applicable |
| water_state | Water conditions | calm, ripples, pronounced waves, fast current, frozen, unknown, not applicable |
| water_color | Water color | turquoise, teal, dark blue, light blue, gray-blue, brown, murky, unknown, not applicable |
| background_density | Building density | no buildings, scattered buildings, moderate building density, dense buildings, unknown, not applicable |
| visibility | Visibility | clear visibility, light haze, visible fog, low visibility, unknown, not applicable |
| wind_level | Visible wind strength | no wind, light breeze, moderate wind, strong wind, very strong wind, unknown, not applicable |
| background_crowd | Background crowd | no people, sparse crowd, moderate crowd, crowded, unknown, not applicable |
| lighting_source | Main light source | natural daylight, moonlight, light through windows, indoor lights, streetlights, neon lights, campfire, mixed light sources, unknown, not applicable |
| lighting_level | Ambient light level | dim, somewhat dim, moderate light, bright, very bright, unknown, not applicable |
| lighting_temperature | Lighting color temperature | cool-toned, neutral-toned, warm-toned, colored lighting, unknown, not applicable |
| lighting_softness | Light quality | hard light, moderately soft light, soft light, diffuse light, unknown, not applicable |
| time_of_day | Time of day | dawn, early morning, morning, noon, afternoon, dusk, night, late night, unknown, not applicable |
| camera_distance | Shot size | long shot, wide shot, medium shot, close shot, unknown, not applicable |

If time_of_day is night or late night, then lighting_source must be moonlight or light through windows or indoor lights or streetlights or neon lights or campfire or mixed light sources or unknown or not applicable.
If vegetation_density is no vegetation, then vegetation_color must be unknown or not applicable.

## Complete examples

Example 1. Input: A traveler has short hair and a beige coat. Other details are unclear.

Output:
```json
{
  "video": "fictional://example-1.mp4",
  "attribute_schema_version": 2,
  "assets": [
    {
      "id": "person",
      "kind": "person",
      "name": "traveler",
      "description": "Traveler",
      "frames": [
        1.0
      ],
      "attributes": {
        "gender_presentation": "unknown",
        "age_group": "unknown",
        "hair_length": "short hair",
        "hair_style": "unknown",
        "hair_color": "unknown",
        "face_shape": "unknown",
        "eye_shape": "unknown",
        "eyebrow_shape": "unknown",
        "nose_shape": "unknown",
        "lip_shape": "unknown",
        "expression": "unknown",
        "makeup_intensity": "unknown",
        "facial_hair": "unknown",
        "height": "unknown",
        "body_build": "unknown",
        "clothing_type": "coat",
        "clothing_color": "beige",
        "clothing_fit": "unknown",
        "accessory": "unknown"
      }
    }
  ],
  "shots": [
    {
      "id": "S1",
      "start": 0.0,
      "end": 2.0,
      "description": "Static shot of the traveler",
      "assets": [
        "person"
      ]
    }
  ],
  "target_order": [
    "S1"
  ]
}
```

Example 2. Input: A blue matte canvas tote bag has a wide strap. Other details are unclear.

Output:
```json
{
  "video": "fictional://example-2.mp4",
  "attribute_schema_version": 2,
  "assets": [
    {
      "id": "object",
      "kind": "object",
      "name": "tote bag",
      "description": "Tote bag with a wide strap",
      "frames": [
        1.0
      ],
      "attributes": {
        "color": "blue",
        "color_lightness": "unknown",
        "color_saturation": "unknown",
        "size": "unknown",
        "shape": "unknown",
        "aspect_ratio": "unknown",
        "material": "canvas",
        "surface_finish": "matte",
        "texture": "unknown",
        "pattern": "unknown",
        "decoration": "unknown",
        "condition": "unknown",
        "transparency": "unknown"
      }
    }
  ],
  "shots": [
    {
      "id": "S1",
      "start": 0.0,
      "end": 2.0,
      "description": "Static shot of the tote bag",
      "assets": [
        "object"
      ]
    }
  ],
  "target_order": [
    "S1"
  ]
}
```

Example 3. Input: A warmly lit coffee shop interior and an alpine lake under clear skies. The exact time and season are unclear.

Output:
```json
{
  "video": "fictional://example-3.mp4",
  "attribute_schema_version": 2,
  "assets": [
    {
      "id": "indoor_environment",
      "kind": "indoor_environment",
      "name": "coffee shop",
      "description": "Wooden tables and glass entrance",
      "frames": [
        1.0
      ],
      "attributes": {
        "room_type": "coffee shop",
        "interior_style": "unknown",
        "space_size": "unknown",
        "ceiling_height": "unknown",
        "layout": "unknown",
        "wall_material": "unknown",
        "floor_material": "unknown",
        "palette": "unknown",
        "furniture_density": "unknown",
        "furniture_style": "unknown",
        "window_type": "unknown",
        "clutter": "unknown",
        "background_crowd": "unknown",
        "lighting_source": "unknown",
        "lighting_level": "unknown",
        "lighting_temperature": "warm-toned",
        "lighting_softness": "unknown",
        "time_of_day": "unknown",
        "camera_distance": "unknown"
      }
    },
    {
      "id": "outdoor_environment",
      "kind": "outdoor_environment",
      "name": "alpine lake",
      "description": "Reflections of distant peaks",
      "frames": [
        3.0
      ],
      "attributes": {
        "setting_type": "lake",
        "geography": "alpine terrain",
        "season": "unknown",
        "weather": "clear skies",
        "surface_material": "unknown",
        "surface_condition": "unknown",
        "vegetation_density": "unknown",
        "vegetation_color": "unknown",
        "water_state": "unknown",
        "water_color": "unknown",
        "background_density": "unknown",
        "visibility": "unknown",
        "wind_level": "unknown",
        "background_crowd": "unknown",
        "lighting_source": "unknown",
        "lighting_level": "unknown",
        "lighting_temperature": "unknown",
        "lighting_softness": "unknown",
        "time_of_day": "unknown",
        "camera_distance": "unknown"
      }
    }
  ],
  "shots": [
    {
      "id": "S1",
      "start": 0.0,
      "end": 2.0,
      "description": "Static shot of the coffee shop",
      "assets": [
        "indoor_environment"
      ]
    },
    {
      "id": "S2",
      "start": 2.0,
      "end": 4.0,
      "description": "Static shot of the alpine lake",
      "assets": [
        "outdoor_environment"
      ]
    }
  ],
  "target_order": [
    "S1",
    "S2"
  ]
}
```
