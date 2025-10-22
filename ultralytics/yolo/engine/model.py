# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path
from typing import Union

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, SegmentationModel, MultiModel,
                                  attempt_load_one_weight, guess_model_task, nn, yaml_model_load)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                                    is_git_dir, yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.yolo.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
# TASK_MAP = {
#     'classify': [
#         ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
#         yolo.v8.classify.ClassificationPredictor],
#     'detect': [
#         DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
#         yolo.v8.detect.DetectionPredictor],
#     'segment': [
#         SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
#         yolo.v8.segment.SegmentationPredictor],
#     'pose': [PoseModel, yolo.v8.pose.PoseTrainer, yolo.v8.pose.PoseValidator, yolo.v8.pose.PosePredictor]}

"""
    这段代码定义了一个名为TASK_MAP的字典，用于将不同任务类型映射到相应的模型和类。具体内容如下：
    'classify': 这个任务类型对应着图像分类任务。在代码中，它包含了ClassificationModel（分类模型）、yolo.v8.classify.ClassificationTrainer（分类训练器）、yolo.v8.classify.ClassificationValidator（分类验证器）以及yolo.v8.classify.ClassificationPredictor（分类预测器）。
    'detect': 这个任务类型是目标检测，其中包含了DetectionModel（检测模型）、yolo.v8.detect.DetectionTrainer（检测训练器）、yolo.v8.detect.DetectionValidator（检测验证器）以及yolo.v8.detect.DetectionPredictor（检测预测器）。
    'segment': 对应着图像分割任务。其中包含了SegmentationModel（分割模型）、yolo.v8.segment.SegmentationTrainer（分割训练器）、yolo.v8.segment.SegmentationValidator（分割验证器）以及yolo.v8.segment.SegmentationPredictor（分割预测器）。
    'pose': 这个任务类型是人体姿势估计，其中包含了PoseModel（姿势模型）、yolo.v8.pose.PoseTrainer（姿势训练器）、yolo.v8.pose.PoseValidator（姿势验证器）以及yolo.v8.pose.PosePredictor（姿势预测器）。
    'multi': 这个类型可能是指多任务学习，其中包含了MultiModel（多任务模型）、yolo.v8.DecSeg.DetectionSegmentationTrainer（检测与分割训练器）、yolo.v8.DecSeg.MultiValidator（多任务验证器）以及yolo.v8.DecSeg.MultiPredictor（多任务预测器）。
"""

TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
        yolo.v8.classify.ClassificationPredictor],
    'detect': [
        DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
        yolo.v8.detect.DetectionPredictor],
    'segment': [
        SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
        yolo.v8.segment.SegmentationPredictor],
    'pose': [PoseModel, yolo.v8.pose.PoseTrainer, yolo.v8.pose.PoseValidator, yolo.v8.pose.PosePredictor],
    'multi': [
        MultiModel, yolo.v8.DecSeg.DetectionSegmentationTrainer, yolo.v8.DecSeg.MultiValidator,
        yolo.v8.DecSeg.MultiPredictor]
}


class YOLO:
    """
    YOLO (You Only Look Once) object detection model.YOLO的目标检测模型类，包括了其构造函数参数、属性和方法。

    Args:
        model (str, Path): Path to the model file to load or create.要加载或创建的模型文件的路径。
        task (Any, optional): Task type for the YOLO model. Defaults to None.YOLO模型的任务类型，默认为None。

    Attributes:属性
        predictor (Any): The predictor object.预测器对象
        model (Any): The model object.模型对象
        trainer (Any): The trainer object.训练器对象
        task (str): The type of model task.模型任务类型
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.如果模型是从 *.pt 文件加载的，这个属性表示检查点对象，通常用于保存和恢复模型的状态信息。
        cfg (str): The model configuration if loaded from *.yaml file.如果模型是从 *.yaml 文件加载的，这个属性表示模型的配置信息，它描述了模型的结构和超参数设置。
        ckpt_path (str): The checkpoint file path.这个属性表示检查点文件的路径，即模型检查点在文件系统中的位置。
        overrides (dict): Overrides for the trainer object.这个属性是用于训练器对象的重写设置，通常可以用来动态地调整训练过程中的参数或行为。
        metrics (Any): The data for metrics. 这个属性包含了模型评估指标的数据，可以用来跟踪模型在训练或测试过程中的性能表现。

    Methods:方法
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method. 这个方法是predict方法的别名，允许像调用函数一样来对实例进行调用，从而触发predict方法。
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.yolo.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.yolo.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.初始化YOLO模型

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        self.callbacks = callbacks.get_default_callbacks()  # 获取默认的回调函数
        self.predictor = None  # reuse predictor，预测器对象，目前为空，重复使用
        self.model = None  # model object，模型对象
        self.trainer = None  # trainer object，训练器对象
        self.task = None  # task type，任务类型
        self.ckpt = None  # if loaded from *.pt，如果从.pt文件加载模型，则表示检查点对象
        self.cfg = None  # if loaded from *.yaml，如果从.yaml文件加载模型，则表示模型配置信息
        self.ckpt_path = None  # 检查点文件路径
        self.overrides = {}  # overrides for trainer object，用于训练器对象的重写设置
        self.metrics = None  # validation/training metrics，用于存储验证/训练指标的数据
        self.session = None  # HUB session，HUB会话对象
        model = str(model).strip()  # strip spaces，将传入的model参数转换为字符串并去除首尾的空格

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):                    # 检查给定的模型是否属于Ultralytics HUB
            from ultralytics.hub.session import HUBTrainingSession  # 导HUBTrainingSession类
            self.session = HUBTrainingSession(model)    # 传入参数model，将新创建的HUBTrainingSession赋值给self.session
            model = self.session.model_file             # 更新model，设为HUBTrainingSession实例的模型文件

        # Load or create new YOLO model，根据给定的模型路径model加载已有的YOLO模型或创建一个新的模型
        suffix = Path(model).suffix         # 获取模型路径的后缀
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:   # 如果模型路径没有后缀且路径的基本名称在GITHUB_ASSET_STEMS中
            model, suffix = Path(model).with_suffix('.pt'), '.pt'   # add suffix添加后缀, i.e. yolov8n -> yolov8n.pt
        if suffix == '.yaml':               # 如果模型的后缀是.yaml
            self._new(model, task)          # 调用方法创建一个新的YOLO模型，并传入模型路径和任务类型task
        else:
            self._load(model, task)         # 加载已有的YOLO模型，并传入模型路径和任务类型task

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model.检查给定的模型是否是Ultralytics HUB的模型"""
        return any((
            model.startswith('https://hub.ultra'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID，检查model分割后的子字符串是否有两个，并且第一个子字符串的长度为42，第二个子字符串的长度为20
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID，检查model的长度是否等于20，使用Path类检查路径model是否存在，字符串model中是否不包含特定字符集‘./\\’中的任何一个字符

    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.
        初始化一个新的模型，并根据模型定义推断任务类型

        Args:
            cfg (str): model configuration file，模型配置文件路径
            task (str) or (None): model task，模型的任务类型，此时为multi,默认为None
            verbose (bool): display model info on load，指定在加载时是否显示显示模型信息，默认为True
        """
        cfg_dict = yaml_model_load(cfg)         # 通过yaml_model_load()函数加载模型配置文件，并返回相应的字典cfg_dict
        self.cfg = cfg                          # 将配置文件路径cfg赋值给self.cfg
        self.task = task or guess_model_task(cfg_dict)  # 推断模型任务类型，如果未指定，会尝试根据cfg_dict的内容猜测任务类型，然后赋值给self.task
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)   # build model，构建模型实例，并将其赋给self.model
        self.overrides['model'] = self.cfg      # 将配置文件路径添加到self.overrides字典中的model键

        # Below added to allow export from yamls，这段代码的目的是允许从YAML文件中导出模型参数
        args = {**DEFAULT_CFG_DICT, **self.overrides}   # 将默认配置字典DEFAULT_CFG_DICT和当前对象的overrides字典合并为一个新的args字典。如果有重复的键名，会优先使用self.overrides中的值。
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 将合并后的args字典中只保留存在于DEFAULT_CFG_KEYS中的键值对，并赋给模型的args属性。这样做可以确保模型只包含指定的关键参数信息。
        self.model.task = self.task                     # 将self.task（任务类型）赋给模型的task属性

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str) or (None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    # 以一个下划线 _ 开头的方法通常被认为是私有方法
    def _check_is_pytorch_model(self):
        """
        Raises TypeError is model is not a PyTorch model
        用于检查模型是否为PyTorch模型。如果模型不是PyTorch模型，则会引发 TypeError 异常。
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'   # 检查模型是否为字符串或路径，并且文件后缀为'.pt'
        pt_module = isinstance(self.model, nn.Module)   # 检查模型是否为PyTorch的nn.Module类型
        if not (pt_module or pt_str):   # 如果模型这两个都不是，则引发TypeError异常
            # 错误消息中还指出PyTorch模型可以用于训练、验证、预测和导出操作，同时提到通过yolo export命令可以导出模型为不同格式，如ONNX、TensorRT，但这些格式只支持predict和val模式
            raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                            f'PyTorch models can be used to train, val, predict and export, i.e. '
                            f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                            f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

    @smart_inference_mode()
    def reset_weights(self):
        """
        Resets the model modules parameters to randomly initialized values, losing all training information.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    @smart_inference_mode()
    def load(self, weights='yolov8n.pt'):
        """
        Transfers parameters with matching names and shapes from 'weights' to model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The tracking results.

        """
        if not hasattr(self.predictor, 'trackers'):
            from ultralytics.tracker import register_tracker
            register_tracker(self, persist)
        # ByteTrack-based method needs low confidence predictions as input
        conf = kwargs.get('conf') or 0.1
        kwargs['conf'] = conf
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.yolo.utils.benchmarks import benchmark
        overrides = self.model.args.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'benchmark'
        overrides = {**DEFAULT_CFG_DICT, **overrides}  # fill in missing overrides keys with defaults
        return benchmark(model=self, imgsz=overrides['imgsz'], half=overrides['half'], device=overrides['device'])

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.接收任意数量的参数
        """
        self._check_is_pytorch_model()          # 检查模型是否为PyTorch模型
        if self.session:                        # 如果存在Ultralytics HUB session会话
            if any(kwargs):                     # 检查是否有传入参数，如果有则警告并忽略本地训练参数，使用HUB训练参数
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()            # 检查Ultralytics包是否有新版本可用，提示用户更新，以获取新功能
        overrides = self.overrides.copy()
        overrides.update(kwargs)                # 把所有传入的参数，以字典形式赋值到overrides中
        if kwargs.get('cfg'):                   # 如果传入了配置文件cfg，则用其覆盖默认参数
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides['mode'] = 'train'             # 设置训练模式
        if not overrides.get('data'):           # 检查数据集是否提供
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):             # 若存在resume参数，则设置为模型的检查点路径，表示从上一次中断的位置开始本次训练
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task  # 设置任务类型，此时为multi多任务
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)   # 初始化训练器对象，根据任务类型选择相应的训练器，根据train时传入的参数和默认回调函数
        if not overrides.get('resume'):  # manually set model only if not resuming              # 如果不是回复训练状态，则设置模型为训练器获取的模型
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml) # 获得一个YOLO Multi多模型对象
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()                     # 开始训练过程，将HUB会话附加到训练器中
        # Update model and cfg after training    # 在训练结束后，更新模型和配置信息，并获取训练结果指标
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self._check_is_pytorch_model()
        self.model.to(device)

    def tune(self,
             data: str,
             space: dict = None,
             grace_period: int = 10,
             gpu_per_trial: int = None,
             max_samples: int = 10,
             train_args: dict = {}):
        """
        Runs hyperparameter tuning using Ray Tune.

        Args:
            data (str): The dataset to run the tuner on.
            space (dict, optional): The hyperparameter search space. Defaults to None.
            grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.
            gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.
            max_samples (int, optional): The maximum number of trials to run. Defaults to 10.
            train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            ModuleNotFoundError: If Ray Tune is not installed.
        """

        try:
            from ultralytics.yolo.utils.tuner import (ASHAScheduler, RunConfig, WandbLoggerCallback, default_space,
                                                      task_metric_map, tune)
        except ImportError:
            raise ModuleNotFoundError("Install Ray Tune: `pip install 'ray[tune]'`")

        try:
            import wandb
            from wandb import __version__  # noqa
        except ImportError:
            wandb = False

        def _tune(config):
            """
            Trains the YOLO model with the specified hyperparameters and additional arguments.

            Args:
                config (dict): A dictionary of hyperparameters to use for training.

            Returns:
                None.
            """
            self._reset_callbacks()
            config.update(train_args)
            self.train(**config)

        if not space:
            LOGGER.warning('WARNING: search space not provided. Using default search space')
            space = default_space

        space['data'] = data

        # Define the trainable function with allocated resources
        trainable_with_resources = tune.with_resources(_tune, {'cpu': 8, 'gpu': gpu_per_trial if gpu_per_trial else 0})

        # Define the ASHA scheduler for hyperparameter search
        asha_scheduler = ASHAScheduler(time_attr='epoch',
                                       metric=task_metric_map[self.task],
                                       mode='max',
                                       max_t=train_args.get('epochs') or 100,
                                       grace_period=grace_period,
                                       reduction_factor=3)

        # Define the callbacks for the hyperparameter search
        tuner_callbacks = [WandbLoggerCallback(project='yolov8_tune')] if wandb else []

        # Create the Ray Tune hyperparameter search tuner
        tuner = tune.Tuner(trainable_with_resources,
                           param_space=space,
                           tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
                           run_config=RunConfig(callbacks=tuner_callbacks, local_dir='./runs'))

        # Run the hyperparameter search
        tuner.fit()

        # Return the results of the hyperparameter search
        return tuner.get_results()

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
