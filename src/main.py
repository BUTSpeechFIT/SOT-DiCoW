import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging

from pretrain_encoder import main as pretrain_encoder
from train import main as train
from train_sot import main as train_sot
from utils.training_args import Cfg, instantiate_arg_classes, process_config

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass
logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg: Cfg = instantiate_arg_classes(cfg)
    process_config(cfg)

    if cfg.training.pretrain_encoder:
        pretrain_encoder(cfg)
    elif cfg.model.mt_sot:
        train_sot(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
