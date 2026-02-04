import abc
import typing

import omegaconf

import clsplats.representation.gaussian_model as gaussian_model

class CLGaussians(gaussian_model.GaussianModel):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self._static_gaussians = {}

    def prune_gaussians(self, typing.Callable[dict, torch.Tensor]) -> None:
        pass

    def unify_gaussians(self) -> None:
        pass

    def split_gaussians(self, active_mask: torch.Tensor) -> None:
        pass