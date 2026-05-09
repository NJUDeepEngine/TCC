from diffusion.model.utils import set_grad_checkpoint


class Registry:
    def __init__(self):
        self.modules = {}

    def register_module(self):
        def decorator(cls):
            self.modules[cls.__name__] = cls
            return cls

        return decorator

    def build(self, cfg, default_args=None):
        cfg = dict(cfg)
        cls_name = cfg.pop("type")
        if cls_name not in self.modules:
            raise KeyError(f"unknown model type: {cls_name}")
        kwargs = dict(default_args or {})
        kwargs.update(cfg)
        return self.modules[cls_name](**kwargs)


MODELS = Registry()


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model
