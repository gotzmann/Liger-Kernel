from torch.nn import CrossEntropyLoss

from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)


class LigerFusedLinearCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerFusedLinearCrossEntropyLoss, self).__init__(*args, **kwargs)

    # gotzmann | https://github.com/linkedin/Liger-Kernel/pull/322/files
    # def forward(self, lin_weight, _input, target, bias=None):
    def forward(self, lin_weight, _input, target, bias=None, reduction=None):
        return LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ignore_index,
            self.label_smoothing,
            self.reduction,
        )
