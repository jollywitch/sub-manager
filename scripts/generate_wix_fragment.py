from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape


def make_id(prefix: str, relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def generate_fragment(input_dir: Path, output_file: Path, component_group: str, directory_ref: str) -> None:
    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No files found under '{input_dir}'.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build the set of all relative directories to recreate the folder tree in MSI.
    dir_paths = {Path(".")}
    for path in files:
        rel_parent = path.relative_to(input_dir).parent
        while True:
            dir_paths.add(rel_parent)
            if rel_parent == Path("."):
                break
            rel_parent = rel_parent.parent

    sorted_dirs = sorted(dir_paths, key=lambda p: (len(p.parts), p.as_posix()))
    dir_id_map: dict[Path, str] = {Path("."): directory_ref}
    for rel_dir in sorted_dirs:
        if rel_dir == Path("."):
            continue
        dir_id_map[rel_dir] = make_id("dir", rel_dir.as_posix())

    children_by_parent: dict[Path, list[Path]] = {}
    for rel_dir in sorted_dirs:
        if rel_dir == Path("."):
            continue
        parent = rel_dir.parent if rel_dir.parent != Path("") else Path(".")
        children_by_parent.setdefault(parent, []).append(rel_dir)
    for key in children_by_parent:
        children_by_parent[key].sort(key=lambda p: p.name.lower())

    files_by_dir: dict[Path, list[Path]] = {}
    for path in files:
        rel = path.relative_to(input_dir)
        parent = rel.parent if rel.parent != Path("") else Path(".")
        files_by_dir.setdefault(parent, []).append(path)
    for key in files_by_dir:
        files_by_dir[key].sort(key=lambda p: p.name.lower())

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">')
    lines.append("  <Fragment>")
    lines.append(f'    <DirectoryRef Id="{escape(directory_ref)}">')

    component_ids: list[str] = []

    def emit_directory(rel_dir: Path, indent: int) -> None:
        pad = " " * indent
        if rel_dir != Path("."):
            dir_id = dir_id_map[rel_dir]
            lines.append(f'{pad}<Directory Id="{dir_id}" Name="{escape(rel_dir.name)}">')
            inner_indent = indent + 2
        else:
            inner_indent = indent

        for file_path in files_by_dir.get(rel_dir, []):
            rel = file_path.relative_to(input_dir).as_posix()
            comp_id = make_id("cmp", rel)
            file_id = make_id("fil", rel)
            source = escape(str(file_path.resolve()))
            lines.append(f'{" " * inner_indent}<Component Id="{comp_id}" Guid="*">')
            lines.append(
                f'{" " * (inner_indent + 2)}<File Id="{file_id}" Name="{escape(file_path.name)}" Source="{source}" KeyPath="yes" />'
            )
            lines.append(f'{" " * inner_indent}</Component>')
            component_ids.append(comp_id)

        for child in children_by_parent.get(rel_dir, []):
            emit_directory(child, inner_indent)

        if rel_dir != Path("."):
            lines.append(f"{pad}</Directory>")

    emit_directory(Path("."), 6)

    lines.append("    </DirectoryRef>")
    lines.append("  </Fragment>")
    lines.append("  <Fragment>")
    lines.append(f'    <ComponentGroup Id="{escape(component_group)}">')
    for comp_id in component_ids:
        lines.append(f'      <ComponentRef Id="{comp_id}" />')
    lines.append("    </ComponentGroup>")
    lines.append("  </Fragment>")
    lines.append("</Wix>")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WiX file/component fragment from a directory.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--component-group", default="AppFiles")
    parser.add_argument("--directory-ref", default="INSTALLFOLDER")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    generate_fragment(
        input_dir=input_dir,
        output_file=output_file,
        component_group=args.component_group,
        directory_ref=args.directory_ref,
    )


if __name__ == "__main__":
    main()
