from . import imports as ip
from . import data_ops as d

def gpu_init(**kwargs):
    # type: () -> None
    """
    Function that selects the GPU's to be used within the Script run. Stores the information of Physical and
    Logilal GPU's that are currently available.
    :return: Nothing
    """
    tf = ip.import_tf()
    kw = kwargs.get
    engine = kw("engine") or 'cpu'
    custom_setup = kw("custom") or False
    if engine == 'cpu':
        ip.os.environ["CUDA_VISIBLE_DEVICES"] = ''
        print('ENGINE: {}'.format(tf.config.experimental.list_physical_devices('CPU')[0]))
    elif engine == 'gpu':
        # GPUs definition, there was a problem when not specified
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                ip.os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                for gpu in gpus:
                    if custom_setup:
                        # Custom setup to prevent memory leaks (hopefully)
                        config = tf.compat.v1.ConfigProto()
                        config.gpu_options.allow_growth = True
                        config.gpu_options.per_process_gpu_memory_fraction = 0.77
                    else:
                        tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                message = str("ENGINE: {} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
                print(message)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        print('ENGINE: Standart CPU setup')
    return


def create_gif(dir_in=None, dir_out=None, name=None, duration=0.5):
    # type: (str, str, str, float) -> None
    """
    Funciton that creates the gif from sequence of pictures.

    :param dir_in: Directory that includes the pictures to be interpreted as gif, default None
    :param dir_out: Desired Directory where to store the created gif, default None
    :param name: Name of the created gif, default None
    :param duration: Gif sequence pause/time during picture switch, default 0.5
    :return: Creates and saves the picture to the desired directory
    """
    filenames = ip.os.listdir(dir_in)
    images = []
    for filename in filenames:
        images.append(ip.imageio.imread("{}\\{}".format(dir_in, filename)))
    ip.imageio.mimsave('{}\\{}.gif'.format(dir_out, name), images, duration=duration, loop=1)
    return


class TbWriter(object):
    """
    Class provides Tensorboard output.
    """
    def __init__(self, logdir):
        """
        Tensorboard writer initialization. creates writers for every Output variable

        :param logdir: Directory in which to save Output
        """
        self._tb_writer_score_legal = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_score_groups = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_score_total = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_score_legal_rm = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_score_groups_rm = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_score_total_rm = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_all_results = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_best_result = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_all_results_rm = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        self._tb_writer_best_result_rm = self.tb_create_writer('{}\\{}'.format(logdir, 'train'))
        pass

    def write(self, **kwargs):
        """
        Function writes data into Output files

        :param it: Sample iteration
        :other params: values in particular iteration
        """
        kw = kwargs.get
        it = kw('it')
        reward_legal = kw('reward_legal')
        reward_group = kw('reward_group')
        reward = kw('reward')
        score_legal_rm = kw('score_legal_rm')
        score_group_rm = kw('score_group_rm')
        score_rm = kw('score_rm')
        img = kw('img')
        img_rm = kw('img_rm')
        best_result = kw('best_result')
        best_result_rm = kw('best_result_rm')

        self.tb_write(wtr=self._tb_writer_score_legal, name='score_legal', val=reward_legal, it=it)
        self.tb_write(wtr=self._tb_writer_score_groups, name='score_groups', val=reward_group, it=it)
        self.tb_write(wtr=self._tb_writer_score_total, name='score_total',
                      val=reward, it=it)
        self.tb_write(wtr=self._tb_writer_score_legal_rm, name='score_legal_rm', val=score_legal_rm,
                      it=it)
        self.tb_write(wtr=self._tb_writer_score_groups_rm, name='score_groups_rm', val=score_group_rm,
                      it=it)
        self.tb_write(wtr=self._tb_writer_score_total_rm, name='score_total_rm',
                      val=score_rm, it=it)
        self.tb_write(wtr=self._tb_writer_all_results, name='all_results', val=img, it=it, typ='image')
        self.tb_write(wtr=self._tb_writer_all_results_rm, name='all_results_rm', val=img_rm, it=it,
                      typ='image')
        self.tb_write(wtr=self._tb_writer_best_result, name='best_result', val=best_result,
                      it=it, typ='image')
        self.tb_write(wtr=self._tb_writer_best_result_rm, name='best_result_rm', val=best_result_rm,
                      it=it, typ='image')
        pass

    @staticmethod
    def tb_create_writer(logdir):
        # type: (str) -> _ResourceSummaryWriter
        """
        Funciton creating the custom TensorBoard writer.
        :param logdir: path of current running file detection
        :return: Writer
        """
        tf = ip.import_tf()
        writer = tf.summary.create_file_writer(
            logdir, max_queue=None, flush_millis=None, filename_suffix=None, name=None,
            experimental_trackable=False)
        return writer

    @staticmethod
    def tb_write(**kwargs):
        # type: (dict) -> None
        """
        TesonrBoard custom writer
        :param kwargs:  wtr - writer, type - Type, name - Name, val - value, it - step
        :return:
        """
        kw, tf = kwargs.get, ip.import_tf()
        val_type, name = kw("typ") or 'scalar', kw("name") or 'undefined'
        wtr, val, it = kw("wtr") or None, kw("val") if kw("val") is not None else 0, kw("it") or 0
        val = kw("val")
        if wtr is None:
            return
        if val is None:
            return
        if val_type == 'scalar':
            with wtr.as_default():
                tf.summary.scalar(name, val, it), wtr.flush()
        elif val_type == 'image':
            with wtr.as_default():
                tf.summary.image(name, val, it), wtr.flush()
        return


def tb_create(folder, name):
    # type: (str, str) -> TensorBoard
    """
    Function that creates the TensorBoard callback
    :return: TensorBoard
    """
    tf = ip.import_tf()
    path = ip.os.getcwd()  # path of current running file detection
    folder = folder or 'run_{}'.format(ip.time.strftime("%m_%d_%H_%M_%S", ip.time.localtime()))
    name = name or 'run_{}'.format(str(ip.np.random.rand())[2:7])
    # --> folder of current analysis
    logdir = str("{}\\runs\\{}\\{}".format(path, folder, name))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=False)
    print("INFO: Tensorboard engine created!")
    return (tensorboard, logdir)


def tb_show():
    # type: () -> None
    """
    Function that opens a browser to show results using TensorBoard.

    *NOTE: When using the fit function in a loop, the TensorBoard stacks all the information into one combined plot,
    thus resulting into bad readable graphic output. It is possible to read the data using the showing method of
    "Relative" when using TensorBoard, but it is rather not suggested.

    *RESOLVED and now fit function stacks all the information into single line.

    :return: Opens a browser with results shown in TensorBoard
    """
    logdir = d.dialog('-d')
    proc = ip.subprocess.Popen("tensorboard --logdir={}".format(logdir))
    ip.webbrowser.open("http://localhost:6006/", new=1)
    out = ip.ctypes.windll.user32.MessageBoxW(0, "To terminate the process, click OK. The process should be terminated "
                                              "after all work wirth tensorboard is done", "TensorBoard running", 1)
    if out == 1:
        proc.terminate()
        return print("INFO: TensorBoard engine closed!")
