from sscompiler.utils.low_quant import LowQuantizer
from sscompiler.utils.high_quant import HighQuantizer
import torch
import unittest


class TestHighQuantizer(unittest.TestCase):

    def setUp(self):
        # This method will run before each test
        self.weights = torch.tensor(
            [[0.1, 0.5, -0.3, 0.9, -1.0, 500], [-166, 0.33, -0.95, -1, 0, 10]],
            dtype=torch.float32,
        )
        self.quantizer = HighQuantizer(bits=8)

    def test_quantize_8bit(self):
        # Test quantization with 8-bit
        self.quantizer.calibrate(self.weights)
        quantized_weights = self.quantizer.quantize(self.weights.unsqueeze(1)).flatten()

        print(quantized_weights)

        # Check if the quantized weights are in the expected range
        self.assertTrue(torch.all(quantized_weights >= -128))
        self.assertTrue(torch.all(quantized_weights <= 127))


if __name__ == "__main__":
    unittest.main()
