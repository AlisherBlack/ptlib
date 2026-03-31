import os
from .defaults import DefaultDataset


class S3DISDataset(DefaultDataset):
    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"
