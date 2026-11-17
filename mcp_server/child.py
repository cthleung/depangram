"""The wrapper a run starts under: it runs python -m engine.regrow or python -m engine.night in this same process and
writes the exit code to a file when the module ends, so run_status can say how a run ended after the server that
started it has gone.

    python -m mcp_server.child EXIT_FILE -m engine.regrow --job NAME ...

The "-m engine.regrow" pair stays on this process's command line on purpose: engine/night.py's busy() looks for it,
so a night or another server sees this run as holding the GPU. A run that is stopped writes no exit code.
"""
import runpy
import sys
import traceback
from pathlib import Path

MODULES = ("engine.regrow", "engine.night")


def exit_code_of(e):
    """The process exit code a SystemExit stands for."""
    if e.code is None:
        return 0
    if isinstance(e.code, int):
        return e.code
    print(e.code, file=sys.stderr)
    return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[1] != "-m" or argv[2] not in MODULES:
        print(f"usage: python -m mcp_server.child EXIT_FILE -m {{{','.join(MODULES)}}} ARGS...", file=sys.stderr)
        return 2
    exit_file, module, rest = Path(argv[0]), argv[2], argv[3:]
    sys.argv = [module, *rest]
    code = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = exit_code_of(e)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    try:
        exit_file.write_text(str(code), encoding="utf-8")
    except OSError:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
