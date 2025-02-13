import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from ..mnts_filters import MNTSFilter
from .intensity_base import MNTSIntensityBase

__all__ = ['N4ITKBiasFieldCorrection']

class N4ITKBiasFieldCorrection(MNTSIntensityBase, MNTSFilter):
    r"""
    Comput the

    Attributes:
        max_iterations (list(int)):
            The maximum number of iterations specified at each fitting level. Default = [50,40,30,20,10]
        convergence_threshold (float):
            Fitting stopping criteria. Default = 0.001
        num_hist_bins (int):
            Number of histogram bins. Default = 200.
    """
    def __init__(self, max_iterations=[50,40,30,20,10]
                 ):
        super(N4ITKBiasFieldCorrection, self).__init__()
        self._max_iterations = max_iterations
        #self._num_of_fitting_lv = 4
        self._convergence_threshold = 0.0001
        self._num_hist_bins = 200

        print(self._max_iterations)

        self._last_bias_field = None
        pass

    @property
    def max_iterations(self) -> List[int]:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val: List[int]) -> None:
        #assert val >= 1, "Number of iteration must be >= 1."
        self._max_iterations = val

    @property
    def num_of_fitting_lv(self) -> int:
        return len(self._max_iterations)

    #@num_of_fitting_lv.setter
    #def num_of_fitting_lv(self, val: int) -> None:
    #    assert val >= 2, "Number of fitting level is meaningless below 2."
    #    self._num_of_fitting_lv = int(val)

    @property
    def convergence_thres(self) -> float:
        return self._convergence_threshold

    @convergence_thres.setter
    def convergence_thres(self, val: float):
        self._convergence_threshold = float(val)

    @ property
    def last_bias_field(self) -> sitk.Image:
        if self._last_bias_field is None:
            self._logger.warning("The bias field was not calculated yet.")
        return self._last_bias_field

    def correct_with_last_bias_field(self, input):
        if self._last_bias_field is None:
            raise ArithmeticError(f"Bias field have not been calculated yet.")
        return input / sitk.Exp(self._last_bias_field)

    def _filter(self,
                input: Union[str, Path, sitk.Image],
                mask: Union[str, Path, sitk.Image] = None):

        input = self.read_image(input)
        mask = self.read_image(mask)

        # Corrector require float input.
        if not input.GetPixelID() in [sitk.sitkFloat32, sitk.sitkFloat64]:
            input = sitk.Cast(input, sitk.sitkFloat32)

        # Shrink input & mask by a factor of 2 to save computational time
        shrinked = sitk.Shrink(input, [2] * input.GetDimension())
        if not mask is None:
            shirinked_mask = sitk.Shrink(mask, [2] * input.GetDimension())


        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        #corrector.SetMaximumNumberOfIterations([self.max_iteration] * self.num_of_fitting_lv)
        corrector.SetMaximumNumberOfIterations(self.max_iterations)
        corrector.Execute(shrinked, shirinked_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(input)
        if not log_bias_field.GetPixelID() == input.GetPixelID():
            log_bias_field = sitk.Cast(log_bias_field, input.GetPixelID())
        self._last_bias_field = log_bias_field
        return input / sitk.Exp(log_bias_field)
