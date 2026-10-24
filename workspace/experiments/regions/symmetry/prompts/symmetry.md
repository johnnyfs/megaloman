You are assessing the intended symmetry of a piece of game-enemy concept art (shown on white).
Classify the design's GLOBAL symmetry as drawn (ignore small shading/highlight differences, but do not
assume symmetry that the drawing does not have):

- "none": no meaningful symmetry
- "vertical": left-right mirror across a vertical axis
- "horizontal": top-bottom mirror across a horizontal axis
- "both": mirror across both a vertical and a horizontal axis
- "radial": n-fold rotational symmetry WITH mirror axes (like a starfish); give the order n
- "rotational": n-fold rotational symmetry WITHOUT mirror axes (like a pinwheel); give the order n

{legend}

Reply with JSON only:
{{"type": "none"|"vertical"|"horizontal"|"both"|"radial"|"rotational",
  "order": <integer n for radial/rotational, else null>,
  "axis_hint": "<where the axis/centre is, e.g. 'centre of the visor', in words>",
  "confidence": <0..1>,
  "per_part": {{"<part name>": {{"symmetric": true|false, "note": "<short: why, esp. if intentionally asymmetric>"}}}},
  "reason": "<one short sentence>"}}
