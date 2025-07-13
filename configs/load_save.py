# load_save_config.py
import os
from omegaconf import OmegaConf
from config import Config

def load_config(yaml_path: str) -> Config:
    """
    Load YAML config file và gán lại giá trị cho đối tượng Config đã khởi tạo sẵn.
    Hỗ trợ các giá trị post_init không bị lỗi.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    
    # Load YAML thành dict
    cfg_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    
    # Khởi tạo config ban đầu (để đảm bảo __post_init__ chạy)
    cfg = Config(dataset_name=cfg_yaml.get('dataset_name', 'CIFAR10'),
                 exp_depths=cfg_yaml.get('exp_depths', [1]),
                 dummy=cfg_yaml.get('dummy', False))

    # Ghi đè các giá trị từ YAML lên config đã khởi tạo
    for section_name, section_val in cfg_yaml.items():
        # Nếu key từ YAML tồn tại trong Config
        if hasattr(cfg, section_name):
            section = getattr(cfg, section_name)

            # Nếu đó là một sub-config (dict) như train, dataset, vgg,...
            if isinstance(section_val, dict):
                for k, v in section_val.items():
                    # Ghi đè từng key nhỏ nếu key tồn tại trong dataclass con
                    if hasattr(section, k):
                        setattr(section, k, v)

            # Nếu là giá trị đơn (dataset_name, dummy, ...)
            else:
                setattr(cfg, section_name, section_val)

    print(f"✅ Config loaded from {yaml_path}")
    return cfg


def save_config(cfg: Config, save_path: str) -> None:
    """
    Lưu toàn bộ config hiện tại sang file .yaml.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cfg_struct = OmegaConf.structured(cfg)
    OmegaConf.save(config=cfg_struct, f=save_path)
    print(f"✅ Config saved to {save_path}")
    