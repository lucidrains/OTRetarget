import pytest
import torch
from ot_retarget import OTRetarget

def test_ot_retarget():
    model = OTRetarget()

    x = torch.randn(2, 3)

    with pytest.raises(NotImplementedError):
        model(x)

if __name__ == '__main__':
    test_ot_retarget()
