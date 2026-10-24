You are advising a pixel artist who shrinks this character from concept art down to a tiny game sprite
(32, 24, 20 and finally 16 px tall). The sprite is built from a REGION MAP: every part is one flat colour.

Images:
1. The concept art.
2. Its full-size region map (each flat colour is one part, see the legend below).
3. A plain proportional shrink of that region map at 64, 32 and 16 px (enlarged, nearest-neighbour).

Legend (colour, name, role, where):
{legend}

The problem: as shapes shrink they lose SIGNIFICANCE. A proportional shrink keeps areas right, but small parts that carry the character stop reading. The art director's notes:
- The brief: a robot snipe redesigned compact to fill a 16x16 square: a round body, a snub-nosed gun for a beak, frog-like folded legs with the feet just under the body. Decide which parts carry that identity at 16 px and how much they need to be exaggerated (into the empty background) to keep it.
- The eyes are fine at proportional size and must NOT be enlarged. They only need a FLOOR: each eye must never shrink
  below a distinct filled rectangle (at least 1x2 px).

The only tools available, applied separately per part:
- "exaggerate": enlarge each piece of the part about an anchor (its centroid, or the point where it attaches to
  its parent part) by a uniform factor up to max_exaggeration, taking pixels from lower-significance parts (and
  from the empty background if allowed). Use this only where a bigger shape is what keeps the character.
- "floor": no enlargement; if the part falls below its minimum footprint at a size, it is replaced there by its
  minimum shape template (e.g. two 1x2 rectangles), and otherwise scales normally.
- "none": scale proportionally. Most parts should be "none": the body masses give up pixels to the significant
  parts. Problems that enlarging cannot fix (merged tentacles, lost gaps) are out of scope: mark those "none".

For EVERY legend part, decide:
- significance: 0.0-1.0, how much of the creature's character this part carries at tiny sizes. When enlarged
  parts collide, the more significant part keeps the pixel. Big body masses are low (they can give pixels away).
- character: one short phrase, what this part contributes to recognising the creature.
- essential_shape: one of "triangle" (a distinct pointed triangular silhouette), "pair_of_rects" (two separate
  rectangles with a gap), "rect", "ellipse", "blob", "stroke". This is the shape each piece must keep.
- mode: "exaggerate" | "floor" | "none".
- max_exaggeration: linear scale factor allowed at the smallest size (1.0 = none, up to 2.5). Only > 1.0 for
  "exaggerate".
- anchor: "centroid" or "attach:<parent part name>" (the piece grows away from where it joins its parent).
- may_claim_background: true if growing outward into empty background is fine (a part on the silhouette that
  should stick out), false if it must stay inside the current outline.
- container: name of the part it must stay inside (e.g. something set inside a frame), or null.
- min_footprint: for sizes "64", "32", "24", "20", "16": the minimum size of EACH separate piece (for paired
  parts: each one of the pair) in sprite pixels: {{"w": int, "h": int, "area": int}}. Use 0s where no minimum
  is needed. Be concrete: what is the smallest pixel shape that still reads as this part's essential shape?
- reason: one sentence.

Return JSON only:
{{"parts": [{{"name", "significance", "character", "essential_shape", "mode", "max_exaggeration", "anchor",
"may_claim_background", "container", "min_footprint", "reason"}} ...], "summary": "<one sentence: what makes
this creature recognisable at 16 px>"}}
