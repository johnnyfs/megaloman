Repaint this image as a flat COLOUR-CODED REGION MAP for a segmentation mask.

Keep the exact same creature, pose, size, position and framing: every part must stay exactly where it is, pixel for
pixel. Change only the colours: fill each part completely with its one flat colour from this table:

- #E6194B  cap (front): The large pointed central mushroom hood fills the upper middle, including all of its painted spots and the red brow immediately above the face.
- #FF66FF  cap_side_lobes (front): The two broad triangular hood lobes flare symmetrically behind the central cap at upper left and upper right.
- #3CB44B  face_housing (front): The pale rounded head casing surrounds the visor across the center, including its screws, tiny blue side lights, and thick lower rim.
- #000075  visor (front): The single dark horizontal window occupies the center of the face housing behind both glowing eyes.
- #FFE119  eyes (front): The two narrow vertical oval eyes sit inside the visor, one left and one right.
- #46F0F0  side_pods (front): The paired angular mechanical pods project from the lower left and lower right sides of the head, including their cyan-lit edges.
- #9A6324  undercarriage (front): The dark segmented mechanical base under the face includes the exposed central chassis, limb-root collars, and small round limb-joint hubs.
- #4363D8  tentacles_front_outer (front): The two thick lateral arms sweep outward from beneath the side pods and curl into the large far-left and far-right bulbous tips.
- #911EB4  tentacles_front_inner (front): The two thick central arms descend from the inner left and inner right roots into the large lower foreground tips, including their embedded cyan lights.
- #0000FF  suction_cups_front (front): The separate ringed suction discs cover the visible ends and inner sides of both outer and both inner foreground tentacles, about ten discs in total.
- #006600  tentacles_back (back): Two narrower curled tentacles descend behind the central foreground pair, visible as segmented pink-and-dark portions on either side of the nozzle; their visible discs are included here.
- #99FF66  nozzle (front): The single short cyan-glowing jet nozzle hangs vertically from the exact center of the undercarriage between the back tentacles.

OCCLUSION LAYERING (important): paint the parts in depth order, back to front:
  1. cap_side_lobes (#FF66FF) - behind cap
  2. cap (#E6194B) - behind face_housing
  3. visor (#000075) - behind eyes
  4. eyes (#FFE119)
  5. side_pods (#46F0F0) - behind face_housing, tentacles_front_outer
  6. tentacles_back (#006600) - behind undercarriage, tentacles_front_inner, nozzle
  7. face_housing (#3CB44B)
  8. undercarriage (#9A6324) - behind face_housing, tentacles_front_outer, tentacles_front_inner, tentacles_back, nozzle
  9. tentacles_front_outer (#4363D8) - behind tentacles_front_inner, suction_cups_front
  10. tentacles_front_inner (#911EB4) - behind suction_cups_front
  11. suction_cups_front (#0000FF)
  12. nozzle (#99FF66)
- A part painted later covers the earlier ones where they overlap. Every FRONT part keeps its complete outline,
  including the curve of each arm where it passes in front of another arm.
- A BACK part shows only where nothing in front covers it; never extend it over a front part.
- Where two front parts of the same material touch (an inner arm over an outer arm), the boundary between their two
  colours must follow the edge of the part in front.

Rules:
- Each part is ONE solid flat colour, exactly the hex value given. No shading, no gradients, no highlights, no
  texture, no spots or stripes (paint pale spots and stripes with the colour of the part they sit on).
- Each suction cup is a solid disc of the cup colour on its arm.
- No outlines, no black lines, no borders between parts: colour regions touch directly.
- Background (everything that is not the creature, including any ground shadow): pure white #FFFFFF.
- No text, no labels, no legend, nothing added. Do not move, resize, redraw or restyle anything; only recolour.
  Nothing inside the creature's outline may stay white.