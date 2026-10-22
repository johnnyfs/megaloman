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
evidence: []
---
Frames of a motion are complementary. When only two frames show a motion, make them its two extremes, with the
most contrast between them. Motion has a direction relative to the way the character faces: if one pose reaches up
and forward, its partner tends down and back (a crouch before a jump leans back when the jump goes up and
forward). Do not make a pose symmetric unless the motion itself is symmetric.
