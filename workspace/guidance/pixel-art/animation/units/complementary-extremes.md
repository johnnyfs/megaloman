---
id: complementary-extremes
scope: pixel-art/animation
applies_to: [animation_frame]
status: hypothesis
provenance:
  author: human
  date: 2026-10-22
  said: "animation frames should have complimentary motion; if a character is running, and we only have two frames to show it, target the most contrast between the two frames; if a character is preparing to jump, if the jumping motion points up and forward, the crouching motion should tend backward"
  prompted_by: games/megaman/classes/small_enemy/animation/pipelines/swim2/versions/v3/runs/squid/s1/feedback.jsonl
evidence:
  - {"run": "games/megaman/classes/small_enemy/animation/pipelines/swim2/versions/v5/runs/squid/s1", "result": "mixed: blessed #2 motion became directional but a mess (human); blessed #1 still symmetric (symmetry 0.72 -> 0.67-0.70). Human: the right extreme for #2 is legs bunched symmetrically under the body, so "avoid symmetric poses" mis-generalizes; the rule should be about opposite extremes of the motion"}
---
Frames of a motion are complementary. When only two frames show a motion, make them its two extremes, with the
most contrast between them. Motion has a direction relative to the way the character faces: if one pose reaches up
and forward, its partner tends down and back (a crouch before a jump leans back when the jump goes up and
forward). Do not make a pose symmetric unless the motion itself is symmetric.
