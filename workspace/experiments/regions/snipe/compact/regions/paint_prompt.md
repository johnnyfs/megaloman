Repaint this image as a flat COLOUR-CODED REGION MAP for a segmentation mask.

Keep the exact same character, pose, size, position and framing: every part must stay exactly where it is, pixel for
pixel. Change only the colours: fill each part completely with its one flat colour from this table:

- #E6194B  body_shell (front): The large round bird-shaped shell fills the centre, including all green, brown, and orange painted body panels but excluding attached features and limbs.
- #008A3B  eye_housing (front): The thick circular dark-rimmed socket sits on the upper-left face around the glowing eye.
- #FFD000  eye (front): The single round glowing eye occupies the centre of the eye housing on the left-facing side.
- #0759D9  upper_beak (front): The long cylindrical tube-like upper beak projects horizontally left from the front of the body, including its broad root collar.
- #66FFFF  lower_beak (front): The short curved lower mandible lies beneath the upper beak and hooks back toward the body.
- #7B00C8  sight (front): The small horizontal telescopic sight and its integral mount sit on top of the body.
- #FF66FF  upper_wing (front): The near wing's large rounded shoulder section occupies the middle-right side of the body, including its orange painted panel.
- #68C400  lower_wing (front): The near wing's green folded outer section curves from the shoulder down and right over the belly, including the small enclosed joint connector.
- #660000  legs (front): This part contains both visible leg instances: the mostly exposed right leg and foot beneath the wing and the separate rear foot protruding from under the body's lower-left edge.
- #000066  tail (front): The single short rounded tail projects up and right from behind the body's rear edge.

OCCLUSION LAYERING (important): paint the parts in depth order, back to front:
  1. eye (#FFD000)
  2. tail (#000066) - behind body_shell
  3. body_shell (#E6194B) - behind eye_housing, upper_beak, lower_beak, sight, upper_wing, lower_wing, legs
  4. eye_housing (#008A3B)
  5. upper_beak (#0759D9)
  6. lower_beak (#66FFFF) - behind upper_beak
  7. sight (#7B00C8)
  8. upper_wing (#FF66FF) - behind lower_wing
  9. lower_wing (#68C400)
  10. legs (#660000) - behind body_shell, lower_wing
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