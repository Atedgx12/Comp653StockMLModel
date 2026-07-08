"""
UCN Hyperparameter Configuration
All tunable hyperparameters in one place as a dataclass.
Fine-tune individual branches by setting frozen_branches and branch_lrs.

Example — fine-tune only the MLP branch on recent data:
    cfg = UCNConfig(
        frozen_branches=('lr', 'nb', 'sent', 'meta'),
        branch_lrs={'mlp': 1e-4},
        epochs=100,
        patience=20,
    )
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class UCNConfig:
    # ── Architecture ─────────────────────────────────────────────────────
    hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    use_sent: bool = True

    # ── Optimizer (Adam, Module 6 Lec 6-5) ───────────────────────────────
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    lam: float = 3e-4          # L2 weight decay

    # ── Training schedule ────────────────────────────────────────────────
    epochs: int = 3000
    batch_size: int = 2048
    warmup_epochs: int = 5     # linear LR warmup before cosine annealing
    patience: int = 150        # early stopping patience
    val_frac: float = 0.15     # temporal validation split fraction

    # ── Regularization ───────────────────────────────────────────────────
    dropout_rate: float = 0.4  # MLP branch inverted dropout
    meta_dropout: float = 0.2  # meta-layer input dropout
    grad_clip: float = 1.0     # global gradient norm clip

    # ── Robustness (Module 8) ────────────────────────────────────────────
    noise_frac: float = 0.02   # Gaussian augmentation: frac of feature std
    use_fgsm: bool = True      # enable PGD adversarial training
    fgsm_eps: float = 0.01     # total PGD epsilon budget
    pgd_steps: int = 5         # PGD inner steps (1 = FGSM)

    # ── Prediction target ────────────────────────────────────────────────
    horizon: int = 1
    # Trading days ahead for the forward-return label.
    # 1=next day, 20=1 month, 63=3 months, 126=6 months

    stride: int = 1
    # Row subsampling step to reduce label autocorrelation.
    # stride=1 keeps every row; stride=horizon keeps only non-overlapping rows.
    # For horizon>20 consider stride=5 to 10 as a compromise.
    # ── Sample weighting ─────────────────────────────────────────────────
    recent_weight_decay: float = 0.0
    # Exponential time-weighting to counter regime shift.
    # 0.0 = uniform, 2.0 = moderate (recommended for long horizons),
    # 4.0 = strong (oldest samples nearly ignored).
    # ── Fine-tuning controls ─────────────────────────────────────────────
    frozen_branches: Tuple[str, ...] = ()
    # Branch keys: 'lr' (logistic reg), 'nb' (naive bayes),
    #              'mlp' (deep MLP),    'sent' (sentiment),
    #              'meta' (meta layer)

    branch_lrs: Dict[str, float] = field(default_factory=dict)
    # Override LR per branch, e.g. {'meta': 1e-4, 'mlp': 5e-4}
    # Branches not listed use the global cfg.lr

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42
    verbose: int = 20          # print every N epochs (0 = silent)

    # ── Branch param key prefixes (used internally) ───────────────────────
    BRANCH_KEYS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        'lr':   ('W_lr', 'b_lr'),
        'nb':   ('mu_nb', 'lsig_nb', 'W_nb', 'b_nb'),
        'mlp':  ('Wm', 'bm'),      # prefix match: Wm1, Wm2, ...
        'sent': ('W_sent', 'b_sent'),
        'meta': ('W_meta', 'b_meta'),
    })

    def param_belongs_to(self, key: str, branch: str) -> bool:
        """Return True if a parameter key belongs to a given branch."""
        prefixes = self.BRANCH_KEYS.get(branch, ())
        return any(key.startswith(p) for p in prefixes)

    def lr_for(self, key: str) -> float:
        """Return the effective learning rate for a parameter key."""
        for branch, branch_lr in self.branch_lrs.items():
            if self.param_belongs_to(key, branch):
                return branch_lr
        return self.lr

    def is_frozen(self, key: str) -> bool:
        """Return True if a parameter key belongs to a frozen branch."""
        return any(self.param_belongs_to(key, b) for b in self.frozen_branches)
