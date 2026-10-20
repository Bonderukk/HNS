"""Modely M1–M5 (predtrénované na ImageNet, knižnica timm)."""
import timm
import torch

MODELS = {
    "M1_ResNet50": "resnet50.a1_in1k",
    "M2_ResNet50_CBAM": "resnet50.a1_in1k",
    "M2b_ResNet50_CBAM_id": "resnet50.a1_in1k",
    "M3_ConvNeXt_T": "convnext_tiny.fb_in22k_ft_in1k",
    "M4_Swin_T": "swin_tiny_patch4_window7_224.ms_in1k",
    "M5_CoAtNet_0": "coatnet_0_rw_224.sw_in1k",
}


class TwoSigmoid(torch.nn.Module):
    """Brána 2·σ(z): pri z = 0 vracia 1, teda CBAM na začiatku nemení príznaky."""
    def forward(self, x):
        return 2 * torch.sigmoid(x)


def cbam_identity_init(model):
    """M2b: CBAM inicializovaný ako identita (nulová posledná vrstva + brána 2·σ)."""
    from timm.layers.cbam import CbamModule
    for m in model.modules():
        if isinstance(m, CbamModule):
            torch.nn.init.zeros_(m.channel.fc2.weight)
            torch.nn.init.zeros_(m.spatial.conv.bn.weight)
            torch.nn.init.zeros_(m.spatial.conv.bn.bias)
            m.channel.gate = TwoSigmoid()
            m.spatial.gate = TwoSigmoid()


def build_model(key: str, num_classes: int, pretrained: bool = True):
    name = MODELS[key]
    if key.startswith("M2"):
        # CBAM (kanálová + priestorová pozornosť) za bn3 v každom Bottleneck bloku, pred reziduálnym súčtom
        model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes,
                                  block_args=dict(attn_layer="cbam"))
        if pretrained:
            src = timm.create_model(name, pretrained=True, num_classes=num_classes).state_dict()
            src = {k: v for k, v in src.items() if not k.startswith("fc.")}
            missing, unexpected = model.load_state_dict(src, strict=False)
            assert not unexpected and all(".se." in k or k.startswith("fc.") for k in missing), missing
        if key == "M2b_ResNet50_CBAM_id":
            cbam_identity_init(model)
        return model
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def head_and_stages(model):
    """Klasifikačná hlava a bloky backbonu od vstupu po výstup (pre postupné rozmrazovanie ResNet50)."""
    head = [model.fc]
    stages = [torch.nn.ModuleList([model.conv1, model.bn1]), model.layer1, model.layer2, model.layer3, model.layer4]
    return head, stages


def gradcam_target(key: str, model):
    """Vrstva pre Grad-CAM a transformácia tvaru výstupu na NCHW."""
    if key.startswith(("M1", "M2")):
        return model.layer4[-1], None
    if key.startswith("M3"):
        return model.stages[-1], None
    if key.startswith("M4"):
        return model.layers[-1].blocks[-1], lambda t: t.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if key.startswith("M5"):
        return model.stages[-1], None
    raise KeyError(key)
