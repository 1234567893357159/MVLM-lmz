from lib.test.utils import TrackerParams
import os
from pathlib import Path


def parameters(yaml_name: str, weight_path: str):
    params = TrackerParams()
    workspace_dir = str(Path(__file__).resolve().parents[3])

    # update default config from yaml file
    yaml_file = os.path.join(workspace_dir, 'experiments', 'mvlm', '%s.yaml' % yaml_name)
    # Each tracker starts from defaults; world settings must not leak into baseline runs.
    from lib.config.mvlm.config import load_config
    local_cfg = load_config(yaml_file)
    params.cfg = local_cfg
    cfg = local_cfg
    params.yaml_name = yaml_name

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = weight_path

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
