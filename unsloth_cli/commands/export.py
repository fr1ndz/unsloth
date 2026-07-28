# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import gc
import hashlib
import logging
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)

EXPORT_FORMATS = [
    "merged-16bit",
    "merged-4bit",
    "gguf",
    "lora",
    "loftq",
    "merged-1bit",
    "nvfp4",
]
GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def list_checkpoints(
    outputs_dir: Path = typer.Option(
        Path("./outputs"), "--outputs-dir", help = "Directory that holds training runs."
    ),
):
    """List checkpoints detected in the outputs directory."""
    from studio.backend.core.export import ExportBackend

    backend = ExportBackend()
    checkpoints = backend.scan_checkpoints(outputs_dir = str(outputs_dir))
    if not checkpoints:
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    for model_name, ckpt_list, metadata in checkpoints:
        typer.echo(f"\n{model_name}:")
        for display, path, loss in ckpt_list:
            loss_str = f" (loss: {loss:.4f})" if loss is not None else ""
            typer.echo(f"  {display}{loss_str}: {path}")


def _cleanup_gpu_memory():
    """Force garbage collection and clear CUDA cache to prevent OOM on sequential exports."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _validate_export(output_path: str) -> bool:
    """Post-export integrity check: verify file exists, is non-empty, and compute checksum."""
    path = Path(output_path)
    if not path.exists():
        logger.error("Validation failed: output path does not exist: %s", output_path)
        return False

    # For directories, check that they contain files
    if path.is_dir():
        files = list(path.rglob("*"))
        if not files:
            logger.error("Validation failed: output directory is empty: %s", output_path)
            return False
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        if total_size == 0:
            logger.error("Validation failed: all files in output directory are empty")
            return False
        typer.echo(f"  ✓ Validated: {len(files)} files, {total_size / 1e9:.2f} GB total")
    else:
        size = path.stat().st_size
        if size == 0:
            logger.error("Validation failed: output file is empty: %s", output_path)
            return False
        # Compute SHA256 for single-file exports
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192 * 1024), b""):
                sha256.update(chunk)
        typer.echo(f"  ✓ Validated: {size / 1e9:.2f} GB, SHA256: {sha256.hexdigest()[:16]}...")

    return True


def export(
    checkpoint: Path = typer.Argument(..., help = "Path to checkpoint directory."),
    output_dir: Path = typer.Argument(..., help = "Directory to save exported model."),
    format: str = typer.Option(
        "merged-16bit",
        "--format",
        "-f",
        help = f"Export format: {', '.join(EXPORT_FORMATS)}",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quantization",
        "-q",
        help = f"GGUF quantization method: {', '.join(GGUF_QUANTS)}",
    ),
    push_to_hub: bool = typer.Option(
        False, "--push-to-hub", help = "Push exported model to HuggingFace Hub."
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help = "HuggingFace repo ID (username/model-name)."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token."
    ),
    private: bool = typer.Option(False, "--private", help = "Make the HuggingFace repo private."),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
    use_loftq: bool = typer.Option(
        False, "--use-loftq", help = "Use LoftQ-aware merge (dequantization-aware weighting)."
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help = "Run post-export integrity check."
    ),
):
    """Export a checkpoint to various formats (merged, GGUF, LoRA adapter)."""
    if format not in EXPORT_FORMATS:
        typer.echo(
            f"Error: Invalid format '{format}'. Choose from: {', '.join(EXPORT_FORMATS)}",
            err = True,
        )
        raise typer.Exit(code = 2)

    if push_to_hub and not repo_id:
        typer.echo("Error: --repo-id required when using --push-to-hub", err = True)
        raise typer.Exit(code = 2)

    from studio.backend.core.export import ExportBackend

    backend = ExportBackend()

    typer.echo(f"Loading checkpoint: {checkpoint}")
    success, message = backend.load_checkpoint(
        checkpoint_path = str(checkpoint),
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)
    typer.echo(message)

    # Disable CUDA graphs before export to prevent stale/frozen weights
    try:
        import torch
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_setGraphEnabled"):
            torch._C._cuda_setGraphEnabled(False)
    except Exception:
        pass

    typer.echo(f"Exporting as {format}{' (LoftQ-aware)' if use_loftq else ''}...")
    output_path: Optional[str] = None
    if format in ("merged-16bit", "loftq"):
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "16-bit (FP16)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
            use_loftq = use_loftq or format == "loftq",
        )
    elif format == "merged-4bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "4-bit (FP4)",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
            use_loftq = use_loftq,
        )
    elif format == "merged-1bit":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "1-bit",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
            use_loftq = use_loftq,
        )
    elif format == "nvfp4":
        success, message, output_path = backend.export_merged_model(
            save_directory = str(output_dir),
            format_type = "NVFP4",
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
            use_loftq = use_loftq,
        )
    elif format == "gguf":
        success, message, output_path = backend.export_gguf(
            save_directory = str(output_dir),
            quantization_method = quantization.upper(),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
        )
    elif format == "lora":
        success, message, output_path = backend.export_lora_adapter(
            save_directory = str(output_dir),
            push_to_hub = push_to_hub,
            repo_id = repo_id,
            hf_token = hf_token,
            private = private,
        )

    # Cleanup GPU memory after export to prevent OOM on sequential exports
    _cleanup_gpu_memory()

    if not success:
        typer.echo(f"Error: {message}", err = True)
        raise typer.Exit(code = 1)

    typer.echo(message)
    if output_path:
        typer.echo(f"Saved to: {output_path}")

        # Post-export validation
        if validate:
            typer.echo("Running post-export validation...")
            if not _validate_export(output_path):
                typer.echo("Error: Post-export validation failed!", err = True)
                raise typer.Exit(code = 3)
