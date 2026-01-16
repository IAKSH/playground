import argparse
import subprocess
import os
import shutil
from pathlib import Path

# Normalize path for set membership
def norm(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


# Parse objdump output
def parse_objdump_output(output):
    deps = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("DLL Name:"):
            dll = line.split(":", 1)[1].strip()
            if dll:
                deps.append(dll)
    return deps


# Check if DLL is inside MinGW root
def is_in_mingw(dll_path, mingw_root):
    return norm(dll_path).startswith(norm(mingw_root))



# Check if DLL is under exe directory tree
def is_in_app_tree(dll_path, exe_dir):
    return norm(dll_path).startswith(norm(exe_dir))


# Resolve a dll path using several search roots
def resolve_dep_path(dll_path, current_dir, exe_dir, mingw_root):
    # absolute path
    if os.path.isabs(dll_path):
        if os.path.isfile(dll_path):
            return os.path.abspath(dll_path)
        return None

    # relative or bare filename
    candidates = [
        os.path.join(current_dir, dll_path),
        os.path.join(exe_dir, dll_path),
        os.path.join(mingw_root, "bin", dll_path),
        os.path.join(mingw_root, dll_path),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return None


# Print SCAN progress (multi-line)
def print_scan(count, filename):
    print(f"{count} files scanned... Current: {filename}", flush=True)


# Recursively collect DLLs
def collect_recursive(objdump, start_file, visited, progress, mingw_root, exe_dir):
    start_file_abs = os.path.abspath(start_file)
    start_file_norm = norm(start_file_abs)

    if start_file_norm in visited:
        return set()

    visited.add(start_file_norm)

    progress["count"] += 1
    print_scan(progress["count"], os.path.basename(start_file_abs))

    # only recurse inside MinGW or app tree
    if not (is_in_mingw(start_file_abs, mingw_root) or is_in_app_tree(start_file_abs, exe_dir)):
        return set()

    # run objdump
    try:
        result = subprocess.run(
            [objdump, "-p", start_file_abs],
            capture_output=True,
            text=True,
            check=False
        )
    except Exception as e:
        print(f"failed to run objdump on {start_file_abs}: {e}")
        return set()

    deps_raw = parse_objdump_output(result.stdout)
    collected = set()
    current_dir = os.path.dirname(start_file_abs)

    for dep in deps_raw:
        resolved = resolve_dep_path(dep, current_dir, exe_dir, mingw_root)
        if not resolved:
            continue

        # collect only MinGW DLLs
        if is_in_mingw(resolved, mingw_root):
            collected.add(norm(resolved))

        # recurse only inside MinGW or app tree
        if is_in_mingw(resolved, mingw_root) or is_in_app_tree(resolved, exe_dir):
            collected.update(
                collect_recursive(objdump, resolved, visited, progress, mingw_root, exe_dir)
            )

    return collected


# Copy DLLs to output dir
def copy_dlls(dlls, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dlls = sorted(set(dlls))
    total = len(dlls)

    if total == 0:
        print("no MinGW DLLs to copy.")
        return

    def print_progress(i, total):
        bar_len = 40
        filled = int(bar_len * (i / total))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\rprogress: [{bar}] {i}/{total}", end="", flush=True)

    print_progress(0, total)

    for i, dll in enumerate(dlls, start=1):
        name = os.path.basename(dll)
        dst = os.path.join(output_dir, name)

        if not os.path.exists(dst):
            shutil.copy2(dll, dst)

        print_progress(i, total)

    print()
    print("DLL copy completed.")


def main():
    parser = argparse.ArgumentParser(description="MinGW objdump recursive DLL collector")
    parser.add_argument("--mingw", required=True, help="Path to MinGW root (contains bin/objdump.exe)")
    parser.add_argument("--input", required=True, help="Executable or DLL to analyze")
    parser.add_argument("--output", required=True, help="Directory to copy DLLs into")

    args = parser.parse_args()

    mingw_root = os.path.abspath(args.mingw)
    objdump = str(Path(mingw_root) / "bin" / "objdump.exe")

    if not os.path.isfile(objdump):
        print(f"objdump not found at: {objdump}")
        return

    start_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    exe_dir = os.path.dirname(start_file)

    print(f"using objdump: {objdump}")
    print(f"analyzing: {start_file}")
    print(f"MinGW root (collect from here): {mingw_root}")
    print(f"app tree root: {exe_dir}")
    print(f"output directory: {output_dir}")

    visited = set()
    progress = {"count": 0}

    dlls = collect_recursive(objdump, start_file, visited, progress, mingw_root, exe_dir)
    print()

    print(f"total MinGW DLLs found: {len(dlls)}")
    for d in sorted(dlls):
        print("  ", d)

    copy_dlls(dlls, output_dir)


if __name__ == "__main__":
    main()
