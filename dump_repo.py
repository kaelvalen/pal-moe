#!/usr/bin/env python3
"""
Dumps repository structure and file contents into a single text file.
Ignores files matched by .gitignore and skips binary files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".gz",
    ".tar",
    ".zip",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".pyc",
    ".so",
    ".dylib",
    ".idx1-ubyte",
    ".idx3-ubyte",
}


def is_binary_file(filepath: Path) -> bool:
    if filepath.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(filepath, "tr") as f:
            f.read(1024)
            return False
    except (UnicodeDecodeError, PermissionError):
        return True


def get_git_files(repo_root: Path) -> list[Path]:
    """Gets all tracked and untracked non-ignored files using git."""
    try:
        cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
        out = subprocess.check_output(cmd, cwd=repo_root, text=True)
        files = []
        for line in out.splitlines():
            line = line.strip()
            if line:
                p = repo_root / line
                if p.is_file():
                    files.append(p)
        return sorted(files)
    except Exception:
        # Fallback if git is not available
        return get_fallback_files(repo_root)


def get_fallback_files(repo_root: Path) -> list[Path]:
    """Simple directory traversal excluding common ignored directories."""
    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "data",
        "dist",
        "build",
    }
    files = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [
            d for d in dirs if d not in ignored_dirs and not d.endswith(".egg-info")
        ]
        for f in filenames:
            p = Path(root) / f
            files.append(p)
    return sorted(files)


def generate_ascii_tree(files: list[Path], repo_root: Path) -> str:
    tree_lines = [f"{repo_root.name}/"]
    paths = [f.relative_to(repo_root) for f in files]

    dirs_seen = set()
    for rel_path in paths:
        parts = rel_path.parts
        for i in range(1, len(parts)):
            sub_dir = Path(*parts[:i])
            if sub_dir not in dirs_seen:
                dirs_seen.add(sub_dir)
                indent = "  " * (len(sub_dir.parts))
                tree_lines.append(f"{indent}├── {sub_dir.name}/")

        indent = "  " * len(parts)
        tree_lines.append(f"{indent}└── {rel_path.name}")

    return "\n".join(tree_lines)


def dump_repository(output_path: Path, repo_root: Path = Path(".")) -> None:
    repo_root = repo_root.resolve()
    files = get_git_files(repo_root)

    # Exclude output file itself from the dump
    files = [f for f in files if f.resolve() != output_path.resolve()]

    print(f"Collecting {len(files)} files from {repo_root}...")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write(f"REPOSITORY DUMP: {repo_root.name}\n")
        out.write("=" * 80 + "\n\n")

        out.write("DIRECTORY HIERARCHY:\n")
        out.write("-" * 80 + "\n")
        out.write(generate_ascii_tree(files, repo_root) + "\n")
        out.write("-" * 80 + "\n\n")

        for f in files:
            rel = f.relative_to(repo_root)
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"FILE: {rel}\n")
            out.write("=" * 80 + "\n\n")

            if is_binary_file(f):
                size_kb = f.stat().st_size / 1024.0
                out.write(f"[BINARY FILE - SKIPPED CONTENT ({size_kb:.1f} KB)]\n")
                continue

            try:
                with open(f, "r", encoding="utf-8", errors="replace") as infile:
                    content = infile.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as e:
                out.write(f"[ERROR READING FILE: {e}]\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Dump written to {output_path} ({size_mb:.2f} MB, {len(files)} files included)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dump entire repository into a single text file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("repo_dump.txt"),
        help="Target output text file (default: repo_dump.txt)",
    )
    args = parser.parse_args()
    dump_repository(args.output)


if __name__ == "__main__":
    main()
