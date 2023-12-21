import numpy as np
from scipy.signal import lfilter, lfilter_zi, butter


class RealTimeFilter:
    def __init__(self, b, a):
        """Initialize the real-time filter with given coefficients.

        Args:
            b (array-like): Numerator coefficients of the filter.
            a (array-like): Denominator coefficients of the filter.
        """
        self.b = b
        self.a = a
        self.zi = lfilter_zi(b, a) * 0  # Initialize filter state

    def process_sample(self, sample):
        """Process a single sample through the filter.

        Args:
            sample (float): The new incoming sample to be filtered.

        Returns:
            float: The filtered sample.
        """
        y, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi, axis=0)
        return y[0]


# Example usage
b, a = butter(N=4, Wn=0.7, btype="low")
rt_filter = RealTimeFilter(b, a)
