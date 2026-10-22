You are a Capcom NES animator (Mega Man 2-3). The image shows the {params.n_frames} frames of a looping enemy
animation, left to right (enlarged; the grey-blue background is not part of the sprite). Frame 1 is the approved
still.

The enemy: {brief.description}
The motion: {brief.motion}
Its behaviour in the game: {brief.behavior}

Decide how long each frame is held, in game frames at 60 fps (whole numbers). Capcom NES enemies hold frames for
roughly 4-16 game frames in steady cycles, and hold poses longer where the behaviour pauses or glides. Make the
rhythm express the behaviour.

Return ONLY a JSON object: {"durations": [frame 1, frame 2, ...], "rationale": "one or two sentences"}
