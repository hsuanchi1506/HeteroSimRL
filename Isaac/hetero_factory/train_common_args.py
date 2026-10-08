"""Command-line arguments shared by the Homo and Hetero training scripts.

Kept apart from train_common.py because the scripts parse arguments before launching the Isaac Sim app, and
train_common.py imports modules that need the app."""


def add_common_args(parser):
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (RULES: 3 / 128 / 4096).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for env and agent.")
    parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default: official config).")
    parser.add_argument("--env_steps", type=int, default=None, help="Env-step budget; sets the PPO iterations.")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Epochs excluded from wall_time_after_warmup.")
    parser.add_argument("--no_sync_timing", action="store_true", help="Do not synchronise the GPU around env.step.")
    parser.add_argument("--grad_every", type=int, default=10,
                        help="Gradient diagnostics on epochs 1-2, then k*N and k*N+1 (0 = off).")
    parser.add_argument("--grad_chunk_seqs", type=int, default=64, help="Sequences per chunk in the gradient probe.")
    parser.add_argument("--grad_max_seqs", type=int, default=None, help="Probe a random subset of this many sequences.")
    parser.add_argument("--collision_stack_size", type=int, default=None,
                        help="PhysX GPU collision stack in bytes. The official 2**28 fits 128 envs only: at 1024 envs "
                             "(nut) and 4096 (every scene) PhysX drops contacts; 2**30 is enough for Hetero at 4096 envs, Homo at 4096 "
                             "(all envs on gear) needs up to 1.45 GB: use 2**31.")
    parser.add_argument("--adv_norm", choices=["global", "per_task", "pooled"], default="global",
                        help="Advantage normalisation: global (rl_games, official) or per_task (analysis experiment: "
                             "each task's samples normalised separately) or pooled (2026-10-07 meeting TODO1: statistics pooled over "
                             "the latest rollout of every task, the Homo counterpart of global). Per-task "
                             "statistics are logged either way.")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save a checkpoint every this many epochs (rl_games save_frequency; official 100), for the "
                             "periodic evaluation and videos of RULES section 8. Not a PPO setting.")
    parser.add_argument("--run_name", type=str, default=None, help="Log directory name.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from this checkpoint.")
    parser.add_argument("--video", action="store_true", default=False, help="Record low-resolution training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of each video (steps).")
    parser.add_argument("--video_interval", type=int, default=20000, help="Steps between videos.")
