import os
import csv
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
from PBN.DomainGeneralization.imcls.src.layers.BatchNorm_v2 import BatchNorm_v2
from PBN.DomainGeneralization.imcls.src.utils.frequencyHelper import generateDataWithDifferentFrequencies_3Channel
from PBN.DomainGeneralization.imcls.src.layers.PBN import *
from PBN.DomainGeneralization.imcls.src.layers.stochnorm import *
from PBN.DomainGeneralization.imcls.src.layers.stochnorm_v2 import *


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, args=None, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            cfg=cfg,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            args=args,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False, aug=False, cl_aug=False):
        f = self.backbone(x, aug=aug, cl_aug=cl_aug)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self, args=None):
        self.args = args
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._optims2 = OrderedDict()  # for clip
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
            self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            assert checkpoint["state_dict"] is not None
            assert checkpoint["epoch"] is not None
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            assert checkpoint["val_result"] is not None
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            # if self.epoch == self.start_epoch:
            #     self.after_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg, args=None):
        super().__init__(args=args)
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_model()
        self.build_data_loader()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        # for hfc
        self.evaluator_low = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.evaluator_high = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.res_base = []
        self.res_low = {12: [], 16: []}
        self.res_high = {12: [], 16: []}

        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_clip = dm.train_loader_clip
        self.train_data = self.train_loader_x.dataset
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        # self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            elif self.cfg.TEST.FINAL_MODEL == "best_val_hfc":
                data = zip(self.res_base, self.res_low[12], self.res_low[16], self.res_high[12], self.res_high[16])
                header = ['base', 'l12', 'l16', 'h12', 'h16']
                output_file = os.path.join(self.output_dir, "bid_cartoon.csv")
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(data)
                print('CSV file has been saved successfully.')
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val_hfc":
            result_base = self.test(split="val_hfc")
            low_result_r12, high_result_r12 = self.test_hfc(split="val_hfc", r=12)
            low_result_r16, high_result_r16 = self.test_hfc(split="val_hfc", r=16)
            self.res_base.append(result_base)
            self.res_low[12].append(low_result_r12)
            self.res_high[12].append(high_result_r12)
            self.res_low[16].append(low_result_r16)
            self.res_high[16].append(high_result_r16)

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_test":
            curr_result = self.test(split="test")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    is_best=True
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None, eval_only=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "val_hfc" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        # for confusion matrix
        # if eval_only:
        #     pltConfusionMatrix(self.evaluator._y_pred, self.evaluator._y_true, self.cfg.DATASET.SOURCE_DOMAINS[0], self.cfg.DATASET.TARGET_DOMAINS[0])

        results = self.evaluator.evaluate()
        print(results)

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_hfc(self, split=None, r=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        self.evaluator_low.reset()
        self.evaluator_high.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "val_hfc" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # for hfc
            input_low, input_high = generateDataWithDifferentFrequencies_3Channel(input.cpu(), r)
            input_low, input_high = input_low.to(label.device), input_high.to(label.device)
            output_low = self.model_inference(input_low)
            output_high = self.model_inference(input_high)
            self.evaluator_low.process(output_low, label)
            self.evaluator_high.process(output_high, label)

        results_low = self.evaluator_low.evaluate()
        results_high = self.evaluator_high.evaluate()
        # print(results_low)
        # print(results_high)

        # for k, v in results.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        return list(results_low.values())[0], list(results_high.values())[0],

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain


class TrainerX_norm(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            # n_iter = self.epoch * self.num_batches + self.batch_idx
            # for name, meter in losses.meters.items():
            #     self.write_scalar("train/" + name, meter.avg, n_iter)
            # self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        if self.cfg.DATASET.NAME == 'ImageNet_DG':
            self.num_classes = 1000
        elif self.cfg.DATASET.NAME == 'PACS':
            self.num_classes = 7
        else:
            AssertionError

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes, args=self.args)

        print("Model_before:", self.model)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        # update model after loading weights
        if cfg.NORM.REPLACE_NORM and not cfg.RESUME:
            TrainerX_norm.update_model(self.model, cfg)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    @classmethod
    def update_model(cls, model, cfg):
        num_norms = cfg.NORM.NUM_NORM
        assert num_norms == 1 or num_norms == 4, 'num_norms != 1 or num_norms != 4'
        if cfg.NORM.NORM_TYPE != 'BN':
            norm_name = cfg.NORM.REPLACE_NORM_NAME
            shallow_name = cfg.NORM.SHALLOW_NAME
            replace_norm = globals()[norm_name] if norm_name else None
            shallow_name = globals()[shallow_name] if shallow_name else None
            apply_layer = cfg.NORM.APPLY_LAYER.split(',')
            shallow_layer = cfg.NORM.SHALLOW_APPLY_LAYER.split(',')
            # split
            apply_layer = list(set(apply_layer) - set(shallow_layer))
            point_group = cfg.NORM.POINT_GROUP
            block_bn_idx = list(map(int, cfg.NORM.BLOCK_BN_IDX.split(',')))
            if shallow_name:
                for i in shallow_layer:
                    if i == '0':
                        try:
                            blockbn_fi = shallow_name(model.module.backbone.bn1, num_norms, point_group=point_group,
                                                      cfg=cfg, block_bn_idx=block_bn_idx)
                            model.module.backbone.__setattr__("bn1", blockbn_fi)
                            # cls._init_BlockBN2d(model.module.backbone.bn1, num_norms, replace_norm, point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                        except:
                            blockbn_fi = shallow_name(model.backbone.bn1, num_norms, point_group=point_group,
                                                      cfg=cfg, block_bn_idx=block_bn_idx)
                            model.backbone.__setattr__("bn1", blockbn_fi)
                            # cls._init_BlockBN2d(model.backbone.bn1, num_norms, replace_norm, point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                    else:
                        layer = 'layer' + i
                        try:
                            cls._init_BlockBN2d(getattr(model.module.backbone, layer), num_norms, shallow_name,
                                                point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                        except:
                            cls._init_BlockBN2d(getattr(model.backbone, layer), num_norms, shallow_name,
                                                point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
            if norm_name:
                for i in apply_layer:
                    if i == '0':
                        try:
                            blockbn_fi = replace_norm(model.module.backbone.bn1, num_norms, point_group=point_group,
                                                      cfg=cfg, block_bn_idx=block_bn_idx)
                            model.module.backbone.__setattr__("bn1", blockbn_fi)
                            # cls._init_BlockBN2d(model.module.backbone.bn1, num_norms, replace_norm, point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                        except:
                            blockbn_fi = replace_norm(model.backbone.bn1, num_norms, point_group=point_group,
                                                      cfg=cfg, block_bn_idx=block_bn_idx)
                            model.backbone.__setattr__("bn1", blockbn_fi)
                            # cls._init_BlockBN2d(model.backbone.bn1, num_norms, replace_norm, point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                    else:
                        layer = 'layer' + i
                        try:
                            cls._init_BlockBN2d(getattr(model.module.backbone, layer), num_norms, replace_norm,
                                                point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                        except:
                            cls._init_BlockBN2d(getattr(model.backbone, layer), num_norms, replace_norm,
                                                point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
            model.to(torch.device(cfg.MODEL.DEVICE))
        print("Model_update:\n{}".format(model))
        return model

    @staticmethod
    def _init_BlockBN2d(model: nn.Module, num_norms, normtype, point_group=1, cfg=None, block_bn_idx=None):
        if isinstance(model, normtype):
            return
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d) \
                    or isinstance(module, StochNorm2d) or isinstance(module, StochNorm1d) \
                    or isinstance(module, StochNorm2d_v2) or isinstance(module, StochNorm1d_v2) \
                    or isinstance(module, BatchNorm_v2):
                BlockBN = normtype(module, num_norms, point_group=point_group, cfg=cfg, block_bn_idx=block_bn_idx)
                model.__setattr__(name, BlockBN)
            else:
                TrainerX_norm._init_BlockBN2d(module, num_norms, normtype, point_group, cfg, block_bn_idx)
