from . import tf
from . import imports as ip
arr, empty = ip.np.array, ip.np.empty


class sequence(tf.keras.utils.Sequence):
    """
    Sequence class used to enhance the training.
    """
    def __init__(self, x_set=empty(0), y_set=empty(0), batch=32):
        self.x, self.y = x_set, y_set
        self.batch = batch

    def __len__(self):
        return int(ip.np.ceil(len(self.x) / self.batch))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]
        return arr(batch_x), arr(batch_y)
