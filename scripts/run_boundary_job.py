"""Detached evaluation wrapper: durable identity, log and completion state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("command required")
    args.state.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "sid": os.getsid(0), "pgid": os.getpgrp(),
             "command": command, "started": time.time(), "exit_code": None}
    def save():
        tmp = args.state / "state.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.state / "state.json")
    save()
    rc = 1
    with (args.state / "task.log").open("w") as log:
        try:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        except Exception as exc:
            log.write(repr(exc) + "\n")
        finally:
            state.update(exit_code=rc, finished=time.time())
            save()
        message = (f"VO境界比較のジョブ終了: {args.state} exit={rc}。"
                   "state.jsonとtask.logおよび結果JSONを確認して報告してください。")
        subprocess.run(["xangi", "tool", "trigger", "--channel", args.channel,
                        "--message", message, "--source", args.state.name],
                       stdout=log, stderr=subprocess.STDOUT)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
