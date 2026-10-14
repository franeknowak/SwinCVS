from scripts.m_swinv2 import SwinTransformerV2, load_pretrained
from scripts.m_swincvs import SwinCVSModel
import torch
import torch.nn as nn

def build_model(config):
    # Initialise Backbone
    model = SwinTransformerV2(  img_size=384, # important to load correct IMNET pretrained weights
                                patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
                                in_chans=config.BACKBONE.SWINV2.IN_CHANS,
                                num_classes=config.BACKBONE.NUM_CLASSES,
                                embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
                                depths=config.BACKBONE.SWINV2.DEPTHS,
                                num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
                                window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
                                mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
                                qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
                                drop_rate=config.BACKBONE.DROP_RATE,
                                drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
                                ape=config.BACKBONE.SWINV2.APE,
                                patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
                                use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
                                pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES)
    
    # Load imagenet weights onto the backbone
    if not config.MODEL.INFERENCE:
        try:
            if 'swinv2_base_patch4' in config.BACKBONE.PRETRAINED:
            # Use imagenet pretrained weights in the backbone
                load_pretrained(config, model)
            elif config.BACKBONE.PRETRAINED is not None:   
                # Use endoscapes pretrained weights in the backbone (for frozen model version)
                print(f"\nLoading backbone weight {config.BACKBONE.PRETRAINED}")
                model.head = nn.Linear(in_features=1024, out_features=3, bias=True)
                weights = 'weights/' + config.BACKBONE.PRETRAINED
                model.load_state_dict(torch.load(weights))
            print(f"Backbone weights loaded successfully!")
        except:
            print("Backbone NOT pretrained!")
    
    if config.MODEL.LSTM:
        # Remove MLP classifier
        model.head = nn.Identity()

        if config.MODEL.E2E != True:
            # Freeze the backbone weights
            for param in model.parameters():
                param.requires_grad = False

        swincvs = SwinCVSModel(model, config)
        return swincvs
    else:
        # Only runs if the pure SwinV2 model option is selected
        model.head = nn.Linear(in_features=1024, out_features=3, bias=True)

    return model


def build_inference_model(config):
    # Initialise Backbone
    model = SwinTransformerV2(
        img_size=384,  # important to load correct IMNET pretrained weights
        patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
        in_chans=config.BACKBONE.SWINV2.IN_CHANS,
        num_classes=config.BACKBONE.NUM_CLASSES,
        embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
        depths=config.BACKBONE.SWINV2.DEPTHS,
        num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
        window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
        mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
        qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
        drop_rate=config.BACKBONE.DROP_RATE,
        drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
        ape=config.BACKBONE.SWINV2.APE,
        patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
        use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
        pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES,
    )

    if not config.MODEL.E2E:
        # Freeze the backbone weights
        for param in model.parameters():
            param.requires_grad = False

    model.head = nn.Identity()

    swincvs = SwinCVSModel(model, config)
    return swincvs

def resolve_weights_file(weights_file):
    """Accept an existing path, or a filename inside ./weights/. Raises if neither exists."""
    from pathlib import Path

    path = Path(weights_file)
    if not path.is_file():
        path = Path("weights") / weights_file
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find weights file '{weights_file}' (looked for it as given, and under ./weights/)."
        )
    return path

def build_finetune_model(config):
    """
    SAFE fine-tuning model: same architecture as `build_inference_model`
    (backbone frozen when MODEL.E2E is False), then initialised from a FULL
    SwinCVS checkpoint (MODEL.INIT_WEIGHTS) with a strict state-dict load.
    Unlike `build_model`, any failure (missing file, missing/unexpected keys,
    shape mismatch) raises instead of silently continuing.
    """
    assert config.MODEL.LSTM, "SAFE fine-tuning only supports SwinCVS (MODEL.LSTM=True)"
    assert not config.MODEL.INFERENCE, "MODEL.INFERENCE must be False for fine-tuning"
    if not config.MODEL.INIT_WEIGHTS:
        raise ValueError("MODEL.INIT_WEIGHTS must point at a full SwinCVS checkpoint for fine-tuning")

    swincvs = build_inference_model(config)

    weights_path = resolve_weights_file(config.MODEL.INIT_WEIGHTS)
    state_dict = torch.load(weights_path, map_location="cpu")
    swincvs.load_state_dict(state_dict, strict=True)  # raises on any key / shape mismatch
    del state_dict
    print(f"Initialised full SwinCVS from '{weights_path}' (strict load OK)")

    n_trainable = sum(p.numel() for p in swincvs.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in swincvs.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")
    return swincvs
