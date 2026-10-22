You are a Capcom NES animator (Mega Man 2-3). The image shows the {params.n_frames} frames available for an enemy,
left to right (enlarged; the grey-blue background is not part of the sprite). Frame 1 is the still.

The enemy: {brief.description}
Its behaviour in the game: {brief.behavior}

Its states: "float" shows the still (frame 1) with no animation; how long it floats is decided by the behaviour,
not by you. You are timing the "{params.state}" state, a {params.mode} action: {params.state_motion}

Design the action as a sequence of frames to show, each with how long it is held in game frames at 60 fps (whole
numbers). The action plays once; afterwards the enemy is back on the still (frame 1), so do not include the float
hold. Capcom NES enemies hold action frames for roughly 4-16 game frames; tells (a pose that warns the player) are
held long enough to read.

Return ONLY a JSON object: {"sequence": [{"frame": n, "duration": n}, ...], "rationale": "one or two sentences"}
