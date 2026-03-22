from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .config import PIPELINE_NAMES, load_config
from .runner import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unified DreamID-V interface across multiple pipeline modes."
    )
    parser.add_argument("--config", required=True, help="Base YAML config file.")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["manual", "express", "dwpose", "faster"],
        choices=sorted(PIPELINE_NAMES),
        help="Pipeline modes to run.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    for pipeline_name in args.pipelines:
        config = copy.deepcopy(base_config)
        config.pipeline.name = pipeline_name
        if pipeline_name != "manual":
            config.inputs.ref_video_facemask = None
            config.inputs.ref_video_pose = None
        if config.output.save_file:
            save_path = Path(config.output.save_file).expanduser().resolve()
            config.output.save_file = str(
                save_path.with_name(f"{save_path.stem}_{pipeline_name}{save_path.suffix}")
            )
        run_inference(config)


if __name__ == "__main__":
    main()
