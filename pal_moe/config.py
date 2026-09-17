"""
JSON config loading, validation and precedence for the experiment runners.

Precedence (highest first): explicit CLI flags > --config values > argparse
defaults. The original inline loader applied config values *after* parsing, so
a config file silently overrode flags typed on the command line; and unknown
keys (e.g. a knob that is not actually wired into the runner, like the old
``lambda_r`` entries in the MNIST configs) were only printed as a warning and
then ignored. Both are errors here.
"""

import json
from typing import Any, Dict

# Numeric knobs that must stay inside a sane domain; (min, max), None = open.
_RANGES: Dict[str, tuple] = {
    "epochs": (1, None),
    "pretrain_epochs": (0, None),
    "feature_dim": (1, None),
    "expert_hidden": (1, None),
    "proto_size": (1, None),
    "proto_samples": (1, None),
    "top_k": (1, None),
    "joint_calib_epochs": (0, None),
    "router_anchor_steps": (0, None),
    "router_anchor_lr": (0.0, None),
    "proto_per_class": (1, None),
    "lambda_ood": (0.0, None),
    "lambda_r": (0.0, None),
    "lambda_e": (0.0, None),
    "proto_routing_alpha": (0.0, 1.0),
}


class ConfigError(ValueError):
    """Invalid --config file: unknown key, wrong type or out-of-range value."""


def load_config(path: str) -> Dict[str, Any]:
    """Reads a JSON config file, requiring a top-level object."""
    with open(path) as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top-level JSON value must be an object")
    return cfg


def _describe(default: Any) -> str:
    if default is None:
        return "int, float, str or bool"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return type(default).__name__


def _type_ok(default: Any, value: Any) -> bool:
    if value is None:
        return default is None
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if default is None:
        return isinstance(value, (bool, int, float, str))
    return isinstance(value, type(default))


def _range_ok(key: str, value: Any) -> bool:
    bounds = _RANGES.get(key)
    if bounds is None or value is None or isinstance(value, bool):
        return True
    lo, hi = bounds
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def apply_config(args: Any, parser: Any, argv: list, path: str) -> None:
    """
    Applies a JSON config to `args` in place.

    Raises ConfigError for unknown keys, type mismatches and out-of-range
    values. Values explicitly passed on the command line (`argv`, including
    ``--key=value`` and ``--no-flag`` forms) win over the config file.
    """
    cfg = load_config(path)
    actions = {a.dest: a for a in parser._actions}

    unknown = sorted(k for k in cfg if k not in actions)
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {unknown}; valid keys: {sorted(actions)}"
        )

    problems = []
    for key, value in cfg.items():
        default = actions[key].default
        if not _type_ok(default, value):
            problems.append(f"{key}={value!r} must be {_describe(default)}")
        elif not _range_ok(key, value):
            lo, hi = _RANGES[key]
            problems.append(f"{key}={value!r} outside allowed range [{lo}, {hi}]")
    if problems:
        raise ConfigError(f"{path}: " + "; ".join(problems))

    # Dests explicitly present on the command line, mapped through the parser's
    # option strings so boolean pairs (--flag / --no-flag) and --key=value work.
    option_to_dest = {
        opt.lstrip("-").replace("-", "_"): action.dest
        for action in parser._actions
        for opt in action.option_strings
    }
    explicit = {
        option_to_dest[normalized]
        for token in argv
        if token.startswith("--")
        and (normalized := token[2:].split("=")[0].replace("-", "_")) in option_to_dest
    }
    cli_values = {dest: getattr(args, dest) for dest in explicit}

    for key, value in cfg.items():
        setattr(args, key, value)
    for dest, value in cli_values.items():
        setattr(args, dest, value)
