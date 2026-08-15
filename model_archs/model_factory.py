"""Model factory for the unified BasicRSISR train/eval entrypoints."""


def normalize_model_type(model_type):
    name = str(model_type or "").strip().lower().replace("-", "_")
    aliases = {
        "atd": "scdr_radr",
        "atd_x4": "scdr_radr",
        "scdr_radr_x4": "scdr_radr",
        "scdr-radr": "scdr_radr",
        "SCDR_RADR": "scdr_radr",
        "ttst_4x": "ttst",
        "hat": "hat_l",
        "hat_l": "hat_l",
        "hsenet_4x": "hsenet",
        "edsr_4x": "edsr",
        "rcan_4x": "rcan",
        "han_4x": "han",
        "haunet_4x": "haunet",
        "nlsa": "nlsn",
        "nlsa_4x": "nlsn",
        "nlsn_4x": "nlsn",
        "transenet_4x": "transenet",
        "ttst_transmamba": "ttst_transmamba",
    }
    return aliases.get(name, name)


def is_scdr_radr_model(model_type):
    return normalize_model_type(model_type) == "scdr_radr"


def build_model_by_type(model_type, scdr_radr_kwargs=None):
    name = normalize_model_type(model_type)
    from utils.basicsr_compat import ensure_minimal_basicsr_compat
    ensure_minimal_basicsr_compat()

    if name == "scdr_radr":
        from utils.model_utils import get_model

        return get_model(**(scdr_radr_kwargs or {}))

    if name == "ttst":
        from model_archs.TTST_arc import TTST
        return TTST()

    if name == "ttst_transmamba":
        from model_archs.TTST_transmamba_arc import TTST_TransMamba
        return TTST_TransMamba()

    if name == "hat_l":
        from model_archs.hat_arch import HAT
        return HAT()

    if name == "hsenet":
        from model_archs.hsenet import HSENET
        return HSENET()

    if name == "edsr":
        from model_archs.edsr import EDSR
        return EDSR()

    if name == "rcan":
        from model_archs.rcan import RCAN
        return RCAN()

    if name == "han":
        from model_archs.han import HAN
        return HAN()

    if name == "haunet":
        from model_archs.haunet import HAUNet
        return HAUNet()

    if name == "nlsn":
        from model_archs.nlsn import NLSN
        return NLSN()

    if name == "transenet":
        from model_archs.transenet import TransENet
        return TransENet()

    raise ValueError("Unsupported --model_type '{}'. Supported: scdr_radr, ttst, ttst_transmamba, hat_l, hsenet, edsr, rcan, han, haunet, nlsn, transenet".format(model_type))
