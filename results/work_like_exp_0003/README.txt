WHAT THIS IS
work_like_exp_0003: benchmark.
Source: data/work_like/experiments/exp_0003 (copied: small result files only).
Model: whisper-large-v3-turbo-ivrit (the real model)
Date: 2026-10-25T01:01:04   Status: completed   Seed: 42
Time estimate for the real model: 4 trials x 20 steps = about 14.8 h (53,324 s) measured with 4 threads at 106 s per step (the slower of the two benchmarks; why is not recorded here). See reports/benchmark.txt.

KEY NUMBERS
Estimated total time: 53324 s (14.8 h). Measured: model=whisper-large-v3-turbo-ivrit, threads=4, seconds_per_step=106.02599999999318.
Sizes: 4 trials x 20 steps, 2 finalists, Val 101 clips, Test 97 clips.

FILES: 16 files, 22 KB. config_full.yaml = the config used; status.json = per-stage state;
logs/run.log.txt = the run log (renamed, .gitignore ignores *.log); reports/ = the stage reports.
NOT HERE: model weights, candidates/, checkpoints/, caches, audio.
TRANSCRIPT TEXT in: none
