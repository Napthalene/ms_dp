from . import imports as ip
from . import mat_ops as m


class opt(object):
    """
    Object for handling the input data.
    """

    def __init__(self, path: str) -> None:
        if path is None:
            raise ValueError("Path is not definied!")
        # --> loading of document
        self.__in = ip.p.read_excel(path)
        pass

    def __repr__(self):
        return self.__in.__repr__()

    def __len__(self):
        return self.__in.__len__()

    def get(self, instance: str, idx=None) -> None:
        if instance == 'all':
            if idx is None:
                return self.__in
            else:
                return self.__in[idx:idx + 1]
        else:
            return self.__in[idx:idx + 1].get(instance).values[0]

    def write(self, instance: str, idx=None, value=0) -> None:
        if self.__in[instance].dtype == 'int64':
            self.__in[instance] = self.__in[instance].astype(float)

        self.__in.at[idx, instance] = value
        return self.__in[idx:idx + 1].get(instance).values[0]


class memory(object):
    """
    Replay memory object
    """
    def __init__(self, **kwargs):
        kw = kwargs.get
        self._in = kwargs
        self._type = kw('type') or 'orm'
        self._batch = kw('batch') or 32
        self._select = kw('selection') or 'rnd'
        self._pos = 0    # position of the latest instance
        self._counter = -1   # dataset offset counter
        self.shape = kw('shape')
        self.mem = ip.np.ndarray(self.shape)

    def __len__(self):
        return self._pos

    def __call__(self):
        """
        Call method of class, returns filled memory.
        :return: Memory instances up to unfilled position
        """
        if self._type == 'frm' and self._select == 'all':
            return self.get_set()
        else:
            return self.mem[:self._pos, :]

    def __getattr__(self, item):
        """
        Get attribute method
        :param item: Instance name to be returned
        :return: Class instance
        """
        if str(item) == 'cells':
            return self.get_cells()
        elif str(item) == 'max_cells':
            return ip.np.max(self.get_cells())
        elif str(item) == 'samples':
            if self._select == 'ru':
                return self.get_samples()
            elif self._select == 'seq':
                return self.get_sequence()
            elif self._select == 'set':
                return self.get_set()
            elif self._select == 'all':
                return self.__call__()
            elif self._select == 'rnd':
                batch = self.check()
                return self.mem[ip.np.random.choice(range(self._pos), batch), :]
        else:
            return self.item

    def append(self, instance):
        """
        Append class method.
        :param instance: instance to be appended
        :return: None
        """
        self.mem[self._pos,:] = instance
        self._pos += 1
        return

    def clear(self):
        """
        Clear class method
        :return: Based on given condition, resets the memory
        """
        if self._type == 'frm':
            del self.mem
            self._pos = 0
            self._counter = -1
            self.mem = ip.np.ndarray(self._in.get('shape'))
        else:
            pass
        return

    def check(self):
        if self._pos >= self._batch:
            return self._batch
        elif self._pos < self._batch:
            return self._pos

    def get_cells(self):
        """
        Method to get number of cells per sample instance saved within class
        :return: Number of cells per sample instance
        """
        return ip.np.sum(~ip.np.isnan(self.mem[:self._pos, :-5]), axis=1)

    def get_samples(self):
        """
        Get samples method, creating dataset
        :return: Dataset (randomly uniformed)
        """
        batch = self.check()
        n_cells, cells, mem = self.shape[1] - 5, ip.np.array(self.cells), self.mem[:self._pos, :].copy()
        samples = batch // n_cells
        counts, counter = list(ip.ct(cells).values()), 0  # counts of specific occurencies
        data = ip.np.ones((1, n_cells + 5))
        for i in range(self.max_cells):
            ids = ip.np.where(cells == i + 1)[0]
            if len(ids) == 0:
                counter += 1
                pass
            else:
                if counts[i - counter] >= samples:  # if there are more desired samples than currently present ones
                    ids = ip.np.random.choice(ids, samples)
                    data = ip.np.concatenate((data, mem[ids]), axis=0)
                    pass
                else:
                    data = ip.np.concatenate((data, mem[ids]), axis=0)
        data = ip.np.delete(data, 0, axis=0)
        lines_to_add = batch - data.shape[0]
        ids = ip.np.random.choice(range(mem.shape[0]), lines_to_add)
        data = ip.np.concatenate((data, mem[ids]), axis=0)
        return data

    def get_sequence(self):
        """
        Get samples method, creating dataset
        :return: Dataset (Sequence oriented)
        """
        batch = self.check()
        # -> get the number of max cells, available samples etc.
        n_cells, cells, mem = self.shape[1] - 5, self.cells, self.mem[:self._pos, :].copy()
        min_cells, max_cells = cells.min(), cells.max()

        # -> split desired batch
        samples = batch // max_cells

        # -> get best samples
        ids = ip.np.where(cells == max_cells)[0]
        ids = ip.np.random.choice(ids, samples)

        # -> recursively run through samples to create sequence
        data = ip.np.ones((1, n_cells + 5))
        for sample in mem[ids, :]:
            action, sample = sample[-5], sample[:-5]
            sample_data = ip.np.tile(sample, (max_cells - min_cells + 1, 1))
            mask = ~ip.np.tri(sample_data.shape[0], sample_data.shape[1], k=min_cells - 1, dtype=bool)
            sample_data[mask] = ip.np.nan
            actions = ip.np.diag(sample_data, k=min_cells - 1)
            actions = ip.np.delete(actions, 0)
            actions = ip.np.append(actions, action)
            sample_data = ip.np.concatenate((sample_data, ip.np.ones((len(actions), 5))), axis=1)
            sample_data[:, -5] = actions
            data = ip.np.concatenate((data, sample_data), axis=0)
        data = ip.np.delete(data, 0, axis=0)
        lines_to_add = batch - data.shape[0]
        ids = ip.np.random.choice(range(mem.shape[0]), lines_to_add)
        data = ip.np.concatenate((data, mem[ids]), axis=0)
        return data

    def get_set(self):
        """
        Function .... to be added
        :return:
        """
        self._counter += 1
        return self.mem[self._counter * self._batch:(self._counter + 1) * self._batch, :]


class BO_df(object):

    def __init__(self, **kwargs):
        kw = kwargs.get
        self.ops = kw('ops')
        self.idx = kw('idx')
        self.n_lines = kw('n_lines')
        type = kw('type') or 'opt'

        if type == 'opt':
            self.__in = ip.p.concat([self.ops._opt__in[self.idx:self.idx + 1] for i in range(self.n_lines)],
                                    ignore_index=True)
        elif type == 'BO_df':
            self.__in = ip.p.concat([self.ops._BO_df__in[i:i+1] for i in self.idx],
                                    ignore_index=True)
        pass
    def get(self, instance: str, idx=None) -> None:
        if instance == 'all':
            if idx is None:
                return self.__in
            else:
                return self.__in[idx:idx + 1]
        else:
            return self.__in[idx:idx + 1].get(instance).values[0]

    def write(self, instance: str, idx=None, value=0, Integer = False) -> None:
        if Integer:
            if self.__in[instance].dtype == 'float64':
                self.__in[instance] = self.__in[instance].astype(int)
        else:
            if self.__in[instance].dtype == 'int64':
                self.__in[instance] = self.__in[instance].astype(float)

        self.__in.at[idx, instance] = value
        return self.__in[idx:idx + 1].get(instance).values[0]

    def get_line_object(self, idx):
        return BO_df(ops=self, idx=idx, n_lines=1, type='BO_df')
