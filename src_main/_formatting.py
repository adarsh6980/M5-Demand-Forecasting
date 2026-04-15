"""
Terminal formatting helpers for the M5 pipeline.
Shared by clean_raw_data.py and run_pipeline.py to avoid duplication.
"""


class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def c(text, color):
    """Wrap text in ANSI color codes."""
    return f"{color}{text}{TermColors.RESET}"


def header(title, emoji="", width=72):
    """Print a prominent section header."""
    print()
    print(c("═" * width, TermColors.CYAN))
    prefix = f"  {emoji}  " if emoji else "  "
    print(c(f"{prefix}{title}", TermColors.BOLD + TermColors.WHITE))
    print(c("═" * width, TermColors.CYAN))


def subheader(title, emoji=""):
    """Print a sub-section header."""
    prefix = f"{emoji} " if emoji else ""
    print(f"\n  {prefix}{c(title, TermColors.BOLD + TermColors.YELLOW)}")


def info(label, value, indent=5):
    """Print a label: value pair."""
    spaces = " " * indent
    print(f"{spaces}{c(label + ':', TermColors.DIM)}  {value}")


def divider(char="─", width=72):
    """Print a thin divider line."""
    print(c(f"  {char * (width - 4)}", TermColors.DIM))


def progress(step, total, message):
    """Print a progress bar with percentage."""
    bar = "█" * step + "░" * (total - step)
    pct = int(step / total * 100)
    print(f"\n  [{c(bar, TermColors.GREEN)}] {c(f'{pct}%', TermColors.BOLD)} {message}")


def write_log(log_file, message):
    """Write a message to the log file if one is open."""
    if log_file:
        log_file.write(message + "\n")
