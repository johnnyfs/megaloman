---
id: still-parts-stay-still
scope: pixel-art/animation
applies_to: [animation_frame]
status: hypothesis
provenance:
  author: human
  date: 2026-10-22
  said: "think about which parts of the body of the animated creature are changing during motion and which aren't; in a typical runner, the legs are moving one way, the arms another, the body rotates in a complementary fashion, but the head is still; parts that don't move should not change, especially metaparts like outlines or expressive parts like eyes"
  prompted_by: games/megaman/classes/small_enemy/animation/pipelines/swim2/versions/v3/runs/squid/s1/feedback.jsonl
evidence: []
---
Before drawing a frame, decide which parts of the body move in this motion and how each one moves (in a typical
run, the legs swing one way, the arms the other, the body rotates in a complementary way, and the head stays still).
Parts that do not move must not change at all: the same pixels in the same place. This matters most for outlines
(keep exactly the same line weight) and for expressive parts like the eyes.
