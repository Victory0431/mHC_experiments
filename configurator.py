"""Small config helper compatible with nanoGPT-style config files.

Usage:
    python train.py configs/vanilla.py --max_iters=200 --batch_size=8
"""

import ast
import sys


for arg in sys.argv[1:]:
    if arg.endswith(".py"):
        print(f"overriding config with {arg}:")
        with open(arg, "r", encoding="utf-8") as f:
            print(f.read())
        exec(open(arg, "r", encoding="utf-8").read())
    elif arg.startswith("--"):
        key, value = arg[2:].split("=", 1)
        if key not in globals():
            raise ValueError(f"unknown config key: {key}")
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        old = globals()[key]
        if isinstance(old, bool) and isinstance(parsed, str):
            parsed = parsed.lower() in {"1", "true", "yes", "on"}
        print(f"overriding: {key} = {parsed}")
        globals()[key] = parsed

