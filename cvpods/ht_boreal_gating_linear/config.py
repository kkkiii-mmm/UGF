from augmentations import StrongAug, WeakAug
from dataset import HPWRENNEMO, BOREAL

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    DATASETS=dict(
        SUPERVISED=[
            (HPWRENNEMO, ),
        ],
        UNSUPERVISED=[
            (BOREAL, ),
        ],
        TEST=("hpwrennemo2boreal_val",),
    ),
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        BACKBONE=dict(NAME='res',),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"],),
        UGF=dict(
            ENABLED=True,
            BETA=2.0,
            BETA_END=0.0,
            BETA_SCHEDULE_STEPS=40000,
            # swap to "cosine" / "exponential" / "sigmoid" / "gaussian_rampup" to change the decay shape;
            # BETA_SCHEDULE_RATE only affects exponential/sigmoid/gaussian_rampup (None = their own default)
            BETA_SCHEDULE_TYPE="gaussian_rampup",
            BETA_SCHEDULE_RATE=None,
            EPS=1.0e-6,
            REFERENCE_LEVEL="P4",
            # one of "normalized_l2" (default) / "raw_variance" / "cosine" -- see ugf.py
            UNCERTAINTY_METHOD="normalized_l2",
            RELIABILITY_ENABLED=False,
            RELIABILITY_ALPHA=1.0,
            UNCERTAINTY_LOSS_ENABLED=True,
            UNCERTAINTY_LOSS_WEIGHT=0.2,
            UNCERTAINTY_LOSS_WEIGHT_START=0.0,
            UNCERTAINTY_LOSS_SCHEDULE_STEPS=40000,
            # swap to "cosine" / "exponential" / "sigmoid" / "gaussian_rampup" to change the ramp shape;
            # UNCERTAINTY_LOSS_SCHEDULE_RATE only affects exponential/sigmoid/gaussian_rampup
            UNCERTAINTY_LOSS_SCHEDULE_TYPE="gaussian_rampup",
            UNCERTAINTY_LOSS_SCHEDULE_RATE=None,
        ),
        FCOS=dict(
            NUM_CLASSES=1,
            QUALITY_BRANCH='iou',
            CENTERNESS_ON_REG=False,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="iou",
            CENTER_SAMPLING_RADIUS=0,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
            VFL=dict(
                USE_SIGMOID=True,
                ALPHA=0.75,
                GAMMA=2.0,
                WEIGHT_TYPE='iou',
                LOSS_WEIGHT=1.0
            ),
        ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=5000,
            STEPS=(8000, ),
            WARMUP_ITERS=250,
            WARMUP_FACTOR=1.0 / 1000,
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=1250,
        CLIP_GRADIENTS=dict(
            ENABLED=True,
            CLIP_TYPE="norm",
            CLIP_VALUE=35.0,
            NORM_TYPE=2.0,
        ),
    ),
    TRAINER=dict(
        NAME="SemiRunner",
        EMA=dict(
            DECAY_FACTOR=0.9999,
            UPDATE_STEPS=1,
            START_STEPS=6000,  # adjust for 4-GPU setting
            FAKE=False
        ),
        SSL=dict(
            BURN_IN_STEPS=10000,  # adjust for 4-GPU setting
        ),
        DISTILL=dict(
            RATIO=0.01,
            HM=dict(
                ALPHA=0.5,
                BETA=0.5
            ),
            SUP_WEIGHT=1,
            UNSUP_WEIGHT=1,
            SUPPRESS='linear',
            WEIGHTS=dict(
                LOGITS=4.,
                DELTAS=1.,
                QUALITY=1.,
                VFL_UNSUP=1.,
                UHL=1.,
            ),
            UN_REGULAR_ALPHA=1.0,
            GAMMA=2.
        ),
        # WINDOW_SIZE=1,
    ),
    TEST=dict(
        EVAL_PERIOD=500,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                SUPERVISED=(WeakAug, dict(short_edge_length=800, max_size=1333, sample_style="choice")),
                UNSUPERVISED=(StrongAug,)
            ),
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR='outputs/ht_only',
    WANDB=True,
    GLOBAL=dict(
        LOG_INTERVAL=10,
    )
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
