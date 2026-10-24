Repaint this image as a flat COLOUR-CODED REGION MAP for a segmentation mask.

Keep the exact same character, pose, size, position and framing: every part must stay exactly where it is, pixel for
pixel. Change only the colours: fill each part completely with its one flat colour from this table:

- #E6194B  body (front): The large rounded torso occupies the central lower half of the character from the chest to the dark belly.
- #3CB44B  head (front): The rounded mechanical head sits above the torso and surrounds the eye at the upper centre-right.
- #911EB4  eye_housing (front): The thick circular socket is inset into the visible left-facing side of the head around the eye.
- #FFD000  eye (front): The single round cream-and-gold eye fills the centre of the circular eye housing.
- #0057D9  tube_beak (front): The long tapered tube projects horizontally left from the head and includes its dark tip.
- #66FFFF  sight (front): The compact cylindrical sight and its mount sit horizontally on top of the head.
- #FF66FF  neck (front): The short ribbed neck connects the underside of the head to the top of the torso.
- #663300  wing (front): The single visible near-side wing covers the right side of the torso, including its round shoulder joint and layered pointed plates.
- #000066  tail_feather_front (front): The foremost lower bronze tail blade projects rightward from beneath the rear edge of the wing.
- #CC9966  tail_feather_back (back): The partially hidden upper and middle tail blades fan upward and right behind the wing and foremost tail blade.
- #00FF99  leg_front (front): The nearer leg descends through the image-right knee and shin to the image-right multi-toed foot, including all its joints and toe pieces.
- #00FF00  leg_back (back): The farther leg is partly hidden behind the nearer leg and descends to the image-left multi-toed foot.

OCCLUSION LAYERING (important): paint the parts in depth order, back to front:
  1. tail_feather_back (#CC9966) - behind wing, tail_feather_front
  2. leg_back (#00FF00) - behind body, leg_front
  3. tube_beak (#0057D9) - behind head
  4. sight (#66FFFF)
  5. neck (#FF66FF) - behind head, body
  6. head (#3CB44B) - behind eye_housing
  7. eye_housing (#911EB4) - behind eye
  8. eye (#FFD000)
  9. tail_feather_front (#000066) - behind wing
  10. leg_front (#00FF99) - behind body
  11. body (#E6194B) - behind wing
  12. wing (#663300)
- A part painted later covers the earlier ones where they overlap. Every FRONT part keeps its complete outline,
  including the curve of each limb where it passes in front of another.
- A BACK part shows only where nothing in front covers it; never extend it over a front part.
- Where two front parts of the same material touch (a near leg over a far leg), the boundary between their two
  colours must follow the edge of the part in front.

Rules:
- Each part is ONE solid flat colour, exactly the hex value given. No shading, no gradients, no highlights, no
  texture, no spots or stripes (paint pale spots and stripes with the colour of the part they sit on).
- No outlines, no black lines, no borders between parts: colour regions touch directly.
- Background (everything that is not the character, including any ground shadow): pure white #FFFFFF.
- No text, no labels, no legend, nothing added. Do not move, resize, redraw or restyle anything; only recolour.
  Nothing inside the character's outline may stay white.