"""
Configuration override helper for training scripts.

This file is executed via `exec(open("configurator.py").read())` from a training script.
It processes command-line arguments in order:

1) A positional argument without `=` is treated as a path to a Python config file. The file is printed
   and executed in the caller's global namespace.
2) Arguments of the form `--name=value` override an existing global variable.

For `--name=value` overrides, values are parsed with `ast.literal_eval` when possible (e.g. numbers,
booleans, lists) and must have the same type as the existing variable.

Note: Config files are arbitrary Python and must be trusted.
"""

import sys
from ast import literal_eval
from collections.abc import Sequence
from tokenize import open as open_source


def _parse_override_value(raw: str) -> object:
    try:
        return literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _exec_config_file(config_file: str, env: dict[str, object]) -> None:
    print(f"Overriding config with {config_file}:")
    with open_source(config_file) as f:
        config_text = f.read()
    print(config_text)
    exec(compile(config_text, config_file, "exec"), env)


def _apply_overrides(argv: Sequence[str], env: dict[str, object]) -> None:
    for arg in argv:
        if "=" not in arg:
            if arg.startswith("--"):
                raise ValueError(f"Expected a config file path, got {arg!r}.")
            _exec_config_file(arg, env)
            continue

        if not arg.startswith("--"):
            raise ValueError(f"Expected an override in the form --name=value, got {arg!r}.")
        key, raw_value = arg[2:].split("=", 1)

        if key not in env:
            raise ValueError(f"Unknown config key: {key}")
        expected_type = type(env[key])
        value = _parse_override_value(raw_value)
        if type(value) is not expected_type:
            raise TypeError(
                f"Type mismatch for {key!r}: expected {expected_type.__name__}, got {type(value).__name__}."
            )

        print(f"Overriding: {key} = {value}")
        env[key] = value


_apply_overrides(sys.argv[1:], globals())
