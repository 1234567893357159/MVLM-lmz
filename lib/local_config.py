import os

class EnvSettings:
    def __init__(self):
        # ── Workspace & output ─────────────────────────────────────────────
        self.workspace_dir       = '/root/MVLM'
        self.tensorboard_dir     = '/root/MVLM/tensorboard'
        self.results_dir         = '/root/MVLM/output/test/tracking_results'
        self.result_plot_dir     = '/root/MVLM/output/test/result_plots'
        self.save_dir            = '/root/MVLM/output'

        # ── Dataset roots (split sub-dirs are handled in each dataset class) ─
        self.lasot_dir       = '/root/MVLM/data/lasot'
        self.tnl2k_dir       = '/root/MVLM/data/tnl2k'
        self.vasttrack_dir   = '/root/MVLM/data/vasttrack'
        self.otb99_lang_dir    = '/root/MVLM/data/otb99_lang'
        self.mgit_dir        = '/Tracker/MVLM/data/mgit'
