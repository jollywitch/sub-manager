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

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">')
    lines.append("  <Fragment>")
    lines.append(f'    <DirectoryRef Id="{escape(directory_ref)}">')

    component_ids: list[str] = []
    for path in files:
        rel = path.relative_to(input_dir).as_posix()
        comp_id = make_id("cmp", rel)
        file_id = make_id("fil", rel)
        source = escape(str(path.resolve()))
        lines.append(f'      <Component Id="{comp_id}" Guid="*">')
        lines.append(f'        <File Id="{file_id}" Source="{source}" KeyPath="yes" />')
        lines.append("      </Component>")
        component_ids.append(comp_id)

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
