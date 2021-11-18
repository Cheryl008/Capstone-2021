import hydra
import os
import subprocess
from omegaconf import OmegaConf, DictConfig

OVERLAY_FILEPATH = "/scratch/wf541/rl_finance_singularity/rl_finance.ext3"
SINGULARITY_IMAGE_FILEPATH = "/scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif"
HYDRA_CONFIG_FILENAME = "hydra_config.yaml"


@hydra.main(config_path=".", config_name=HYDRA_CONFIG_FILENAME)
def main(cfg: DictConfig):
    print(f"Hydra driver script, NOT running in Singularity, calling from cwd = {os.getcwd()}")

    hydra_config_filepath = os.path.join(os.getcwd(), HYDRA_CONFIG_FILENAME)

    with open(hydra_config_filepath, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    subprocess.run(
        [
            "singularity", "exec",
            "--overlay", f"{OVERLAY_FILEPATH}:ro",
            SINGULARITY_IMAGE_FILEPATH,
            "/bin/bash", "-c", f"source /ext3/env.sh; python /scratch/wf541/Capstone-2021/train_ddpg.py {hydra_config_filepath}"
        ]
    )


if __name__ == '__main__':
    main()
