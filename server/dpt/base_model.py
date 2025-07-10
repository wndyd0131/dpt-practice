# This file contains code adapted from the DPT repository:
# https://github.com/intel-isl/DPT
# Licensed under the MIT License.
# Copyright (c) 2021 Intel Corporation

import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)