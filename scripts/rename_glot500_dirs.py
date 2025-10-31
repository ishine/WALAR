#!/usr/bin/env python3
"""
Rename folders under /mnt/gemini/data1/yifengliu/data/glot500 to two-letter ISO codes
when a mapping is available in code/utils.py. If no mapping is available, the folder
will be left unchanged.

Usage:
  python scripts/rename_glot500_dirs.py --dry-run
  python scripts/rename_glot500_dirs.py --apply
"""
import os
import argparse
import importlib.util
import sys

ROOT = "/mnt/gemini/data1/yifengliu/data/glot500"

# Import mappings from code/utils.py by path
UTILS_PATH = os.path.join(os.path.dirname(__file__), '..', 'code', 'utils.py')
UTILS_PATH = os.path.abspath(UTILS_PATH)

spec = importlib.util.spec_from_file_location('utils_for_rename', UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# We will use these mappings if available
three2two = getattr(utils, 'three2two', {})
two2three = getattr(utils, 'two2three', {})
mm_dict = getattr(utils, 'mm_dict', {})
lang_dict = getattr(utils, 'lang_dict', {})
long2lang = getattr(utils, 'long2lang', {})


def infer_two_char(name):
    """Try to infer a two-letter ISO code from a folder name.

    Heuristics implemented:
    - If name is already 2 letters and exists in two2three or mm_dict, keep it.
    - If name contains an underscore like 'eng_Latn' or 'hye_Armn', take the prefix
      (e.g., 'eng') and map via three2two if possible.
    - If name looks like a three-letter code present in three2two, map to two-letter.
    - If name equals a key in mm_dict (already two-letter), accept it.
    - If name matches a key in long2lang (long script-tagged keys), try to map to a
      language name and then to a two-letter using mm_dict or inverted lookup.

    Returns new_name (two-letter) or None if no mapping.
    """
    # already 2 letters
    if len(name) == 2 and name in two2three:
        return name
    # common case: something like eng_Latn, hye_Armn, zho_Hani, etc.
    if '_' in name:
        prefix = name.split('_')[0]
        # prefix may be 3-letter
        if prefix in three2two:
            return three2two[prefix]
        # prefix might already be two-letter
        if prefix in two2three:
            return prefix
        # full name might be in long2lang (e.g., eng_Latn)
        if name in long2lang:
            longname = long2lang[name]
            # longname might be e.g. 'English' -> find in mm_dict values
            for k, v in mm_dict.items():
                if v.lower() == longname.lower():
                    return k
    # if name is a 3-letter code
    if name in three2two:
        return three2two[name]
    # if name already in mm_dict keys (two-letter)
    if name in mm_dict:
        return name
    # try matching language full name -> two-letter
    # check if name (case-insensitive) matches a value in lang_dict or mm_dict
    nlow = name.lower()
    for k, v in mm_dict.items():
        if v.lower().replace(' ', '_') == nlow or v.lower() == nlow:
            return k
    for k, v in lang_dict.items():
        if v.lower().replace(' ', '_') == nlow or v.lower() == nlow:
            # three-letter k -> convert
            if k in three2two:
                return three2two[k]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Actually perform renames')
    parser.add_argument('--force', action='store_true', help='Force overwrite/merge when target exists')
    parser.add_argument('--root', default=ROOT, help='Root directory to operate on')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"Root path not found: {root}")
        sys.exit(1)

    entries = sorted(os.listdir(root))
    rename_map = []
    for ent in entries:
        # skip lock files that start with _mnt_... .lock
        if ent.startswith('_mnt_gemini') or ent.endswith('.lock'):
            continue
        src_path = os.path.join(root, ent)
        if not os.path.isdir(src_path):
            continue
        new_name = infer_two_char(ent)
        if new_name and new_name != ent:
            tgt_path = os.path.join(root, new_name)
            # handle existing target
            if os.path.exists(tgt_path):
                if args.force:
                    print(f"Target exists and will be merged/overwritten due to --force: {ent} -> {new_name}")
                    rename_map.append((ent, new_name))
                else:
                    print(f"WARNING: target exists, skipping: {ent} -> {new_name}")
                    continue
            else:
                rename_map.append((ent, new_name))

    if not rename_map:
        print("No directories to rename based on available mappings.")
        # but we may still want to rename files inside two-letter dirs
        # collect two-letter dirs and plan file renames for them
        rename_map = []

    print("Planned renames:")
    for a, b in rename_map:
        print(f"  {a} -> {b}")

    # Now plan file renames inside directories (for both renamed dirs and existing two-letter dirs)
    # Policy: if a directory will be renamed (old -> new), rename files inside that
    # directory to "<two-letter-code><ext>" (e.g. ara_Arab.jsonl -> ar.jsonl).
    # file_rename_map entries: (src_path, target_path, will_overwrite_bool)
    file_rename_map = []
    # build set of target dirs to operate on: include new names from rename_map
    target_dirs = set(new for _, new in rename_map)
    # also include existing two-letter directories under root
    for ent in entries:
        if len(ent) == 2 and os.path.isdir(os.path.join(root, ent)):
            target_dirs.add(ent)

    for ent in list(target_dirs):
        new_ent = ent
        dir_path = os.path.join(root, ent)
        if not os.path.isdir(dir_path):
            dir_path = os.path.join(root, new_ent)
            if not os.path.isdir(dir_path):
                continue
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if os.path.isdir(fpath):
                continue
            base, ext = os.path.splitext(fname)
            # new filename = two-letter code + ext
            new_fname = new_ent + ext
            target_dir = os.path.join(root, new_ent)
            target_path = os.path.join(target_dir, new_fname)
            # decide overwrite behavior
            if os.path.exists(target_path):
                if args.force:
                    file_rename_map.append((fpath, target_path, True))
                else:
                    print(f"WARNING: file target exists, skipping: {fpath} -> {target_path}")
                    continue
            else:
                file_rename_map.append((fpath, target_path, False))

    if file_rename_map:
        print('\nPlanned file renames inside directories:')
        for a, b, _ in file_rename_map:
            print(f"  {a} -> {b}")

    if args.apply:
        import shutil
        def merge_dirs(src_dir, dst_dir):
            # Move contents of src_dir into dst_dir, overwriting files if needed
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dst_dir, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        # recursively merge
                        merge_dirs(s, d)
                        try:
                            os.rmdir(s)
                        except OSError:
                            pass
                    else:
                        shutil.move(s, d)
                else:
                    # file
                    if os.path.exists(d):
                        os.remove(d)
                    shutil.move(s, d)

        for a, b in rename_map:
            a_path = os.path.join(root, a)
            b_path = os.path.join(root, b)
            if os.path.exists(b_path):
                # target exists
                if args.force:
                    print(f"Merging {a_path} -> {b_path} (target exists, --force)")
                    if not os.path.isdir(b_path):
                        # target exists but is a file; remove it then move dir
                        os.remove(b_path)
                        shutil.move(a_path, b_path)
                    else:
                        # move contents from a_path into b_path
                        merge_dirs(a_path, b_path)
                        # attempt to remove source dir if empty
                        try:
                            os.rmdir(a_path)
                        except OSError:
                            pass
                else:
                    print(f"WARNING: target exists, skipping dir rename: {a_path} -> {b_path}")
            else:
                print(f"Renaming {a_path} -> {b_path}")
                shutil.move(a_path, b_path)
        # perform file renames after directories moved
        # build a map for directory renames for path adjustments
        dir_map = {os.path.join(root, old): os.path.join(root, new) for old, new in rename_map}
        for a, b, will_overwrite in file_rename_map:
            src = a
            tgt = b
            # if original src no longer exists because parent dir was renamed, adjust
            if not os.path.exists(src):
                for old_parent, new_parent in dir_map.items():
                    if src.startswith(old_parent + os.sep):
                        src = src.replace(old_parent, new_parent, 1)
                        break
            # likewise, if target parent doesn't exist, create it
            tgt_parent = os.path.dirname(tgt)
            if not os.path.isdir(tgt_parent):
                os.makedirs(tgt_parent, exist_ok=True)
            if not os.path.exists(src):
                print(f"WARNING: source file not found, skipping: {a} (resolved to {src}) -> {tgt}")
                continue
            # resolve absolute paths
            src_abs = os.path.abspath(src)
            tgt_abs = os.path.abspath(tgt)
            # if same path, nothing to do
            if src_abs == tgt_abs:
                print(f"Source and target are the same, skipping: {src}")
                continue
            if os.path.exists(tgt):
                if args.force or will_overwrite:
                    # ensure we are not deleting the source
                    if tgt_abs == src_abs:
                        print(f"Source and target are the same (after resolve), skipping: {src}")
                        continue
                    print(f"Overwriting file {tgt} with {src}")
                    try:
                        os.remove(tgt)
                    except OSError:
                        pass
                else:
                    print(f"WARNING: target exists, skipping file: {src} -> {tgt}")
                    continue
            # final existence check for source
            if not os.path.exists(src):
                print(f"WARNING: source file not found, skipping final step: {src} -> {tgt}")
                continue
            print(f"Renaming file {src} -> {tgt}")
            os.rename(src, tgt)
        print("Done.")
    else:
        print("Dry-run: no changes made. Re-run with --apply to execute.")


if __name__ == '__main__':
    main()
