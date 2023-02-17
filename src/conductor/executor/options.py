class ExecutionOptions(object):
    _name = "execution_options"

    def __repr__(self):
        return ExecutionOptions._name

    @staticmethod
    def validate(_dict):
        if "device_type" in _dict:
            assert _dict["device_type"] in ["cuda", "cpu"], "device_type must be <cuda, cpu> for execution options"
        if "device_id" in _dict:
            assert isinstance(_dict["device_id"], int), "device_id must be int for execution options"
        if "fill_mode" in _dict:
            assert _dict["fill_mode"] in ["random", "zeros", "ones"], "fill_mode must be <random, zeros, ones> for execution options"
    @staticmethod
    def from_dict(_dict):
        return ExecutionOptions(**_dict)

    def __init__(self, device_type="cuda", device_id=0, fill_mode="random"):
        self.device_type = device_type
        self.device_id = device_id
        self.fill_mode = fill_mode