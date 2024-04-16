import logging
import imports as ip


def import_tf() -> Tensorflow:
    """
    Function that imports Tensorflow only if neccessary to do so.
    :return: Initialized Tensorflow
    """
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    return tf


def tb_create(folder: str, name: str) -> TensorBoard:
    """
    Function that creates the TensorBoard callback
    :returns: TensorBoard Class
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


def tb_show() -> None:
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
    out = ip.ctypes.windll.user32.MessageBoxW(0, "To terminate the process, click OK. The process "
                                                 "should be terminated"
                                              "after all work wirth tensorboard is done",
                                              "TensorBoard running", 1)
    if out == 1:
        proc.terminate()
        return print("INFO: TensorBoard engine closed!")