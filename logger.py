import time
import os
import traceback

LOG_FILE = "highlight.log.txt"
LOG_LEVEL = "DEBUG"  # DEBUG / INFO / WARN / ERROR

# 清空前一次日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Highlight Log ===\n")


def log(tag, msg, level="INFO"):
    now = time.strftime("%H:%M:%S")
    caller = traceback.extract_stack()[-2]
    file = os.path.basename(caller.filename)
    line = caller.lineno

    text = f"[{now}] [{level:5}] [{tag:10}] {msg}  @ {file}:{line}"

    print(text, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def debug(tag, msg):
    if LOG_LEVEL in ("DEBUG",):
        log(tag, msg, "DEBUG")


def info(tag, msg):
    log(tag, msg, "INFO")


def warn(tag, msg):
    log(tag, msg, "WARN")


def error(tag, msg):
    log(tag, msg, "ERROR")