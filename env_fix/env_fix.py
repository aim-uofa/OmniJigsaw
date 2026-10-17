import ast
import io
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(".")
SITE_PACKAGES = PROJECT_ROOT / ".venv/lib/python3.12/site-packages"


class PatchError(RuntimeError):
    """The dependency does not have the expected patchable structure."""


def same_code(node, expected):
    """Compare syntax without source locations, formatting, or comments."""
    return ast.dump(node) == ast.dump(expected)


def only_match(nodes, description):
    matches = list(nodes)
    if len(matches) != 1:
        raise PatchError(f"Expected one {description}; found {len(matches)}.")
    return matches[0]


def find_method(source, class_name, method_name):
    tree = ast.parse(source)
    cls = only_match(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        f"class {class_name}",
    )
    return only_match(
        (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        f"method {class_name}.{method_name}",
    )


def replace_statement(source, node, replacement):
    """Replace a standalone statement while preserving surrounding source."""
    lines = io.StringIO(source, newline="").readlines()
    first, last = node.lineno - 1, node.end_lineno - 1
    # AST column offsets count UTF-8 bytes, not Unicode characters.
    indentation = lines[first].encode("utf-8")[:node.col_offset].decode("utf-8")
    suffix = lines[last].encode("utf-8")[node.end_col_offset:].decode("utf-8").strip()
    if indentation.strip() or (suffix and not suffix.startswith("#")):
        raise PatchError(f"Expected a standalone statement at line {node.lineno}.")
    ending = lines[last][len(lines[last].rstrip("\r\n")):]
    lines[first:last + 1] = [indentation + replacement + ending]
    return "".join(lines)


def patch_video(source):
    """Copy NumPy video arrays before converting them to tensors."""
    method = find_method(source, "BaseVideoProcessor", "_prepare_input_videos")
    original = ast.parse("video = torch.from_numpy(video).contiguous()").body[0]
    replacement = "video = torch.from_numpy(video.copy()).contiguous()"
    patched = ast.parse(replacement).body[0]
    target = only_match(
        (
            node for node in ast.walk(method)
            if isinstance(node, ast.Assign) and (same_code(node, original) or same_code(node, patched))
        ),
        "NumPy video conversion",
    )
    if same_code(target, patched):
        return source
    return replace_statement(source, target, replacement)


def patch_audio_padding(source):
    """Keep the caller's audio padding setting in the Qwen Omni processor."""
    method = find_method(source, "Qwen3OmniMoeProcessor", "__call__")
    condition = ast.parse("audio is not None", mode="eval").body
    branch = only_match(
        (node for node in method.body if isinstance(node, ast.If) and same_code(node.test, condition)),
        "audio input branch",
    )
    extraction = ast.parse('audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])').body[0]
    call = only_match(
        (node for node in ast.walk(method) if isinstance(node, ast.Assign) and same_code(node, extraction)),
        "audio feature extraction",
    )
    assignment = ast.parse('output_kwargs["audio_kwargs"]["padding"] = True').body[0]
    padding = assignment.targets[0]
    writes = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Subscript)
        and not isinstance(node.ctx, ast.Load)
        and same_code(node.value, padding.value)
        and same_code(node.slice, padding.slice)
    ]

    # An absent override is valid only at the expected feature-extraction site.
    if branch.body[0] is call and not writes:
        return source
    if (
        len(branch.body) < 2
        or not same_code(branch.body[0], assignment)
        or branch.body[1] is not call
        or len(writes) != 1
    ):
        raise PatchError("Unexpected audio padding override or feature-extraction layout.")
    return replace_statement(source, branch.body[0], '# output_kwargs["audio_kwargs"]["padding"] = True')


def apply_patches():
    if not SITE_PACKAGES.is_dir():
        raise PatchError(
            f"Python 3.12 site-packages directory not found: {SITE_PACKAGES}. "
            "Run this script from the project root after creating the .venv environment."
        )

    patches = (
        ("transformers/video_processing_utils.py", patch_video),
        ("transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py", patch_audio_padding),
        ("transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py", patch_audio_padding),
    )
    prepared = []
    # Validate every target before writing any dependency files.
    for relative, patch in patches:
        path = SITE_PACKAGES / relative
        original = path.read_bytes()
        try:
            updated = patch(original.decode("utf-8"))
            compile(updated, str(path), "exec", dont_inherit=True)
            if patch(updated) != updated:
                raise PatchError("Patch verification failed.")
        except (PatchError, SyntaxError, UnicodeError) as exc:
            raise PatchError(f"{path}: {exc}") from exc
        prepared.append((path, original, updated.encode("utf-8")))

    audio_source = PROJECT_ROOT / "env_fix/audio_process.py"
    audio_target = SITE_PACKAGES / "qwen_omni_utils/v2_5/audio_process.py"
    audio_bytes = audio_source.read_bytes()
    compile(audio_bytes, str(audio_source), "exec", dont_inherit=True)
    if not audio_target.parent.is_dir():
        raise PatchError(f"Target directory not found: {audio_target.parent}")
    if audio_target.exists() and not audio_target.is_file():
        raise PatchError(f"Expected a dependency file: {audio_target}")

    for path, original, updated in prepared:
        if original != updated:
            path.write_bytes(updated)
            print(f"Patched {path.relative_to(SITE_PACKAGES)}")
        else:
            print(f"Skipped {path.relative_to(SITE_PACKAGES)}: patch already applied")

    # Install the same project-provided PyAV audio loader as before.
    shutil.copy2(audio_source, audio_target)
    print(f"Replaced {audio_target.relative_to(SITE_PACKAGES)} with {audio_source}")

    for path, _, updated in prepared:
        if path.read_bytes() != updated:
            raise PatchError(f"Written patch verification failed: {path}")
    if audio_target.read_bytes() != audio_bytes:
        raise PatchError(f"Audio loader verification failed: {audio_target}")


def main():
    print("Applying dependency compatibility patches...")
    print("-" * 50)
    try:
        apply_patches()
    except (PatchError, OSError, SyntaxError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("-" * 50)
    print("Dependency compatibility patches completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
