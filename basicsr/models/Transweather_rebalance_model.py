import os
import time
from collections import OrderedDict
from os import path as osp
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.nn import functional as F

from basicsr.utils.dist_util import get_dist_info
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

from basicsr.archs import build_network
from basicsr.losses import build_loss, Rebalance_L1


@MODEL_REGISTRY.register()
class IRModel_TransWeather_Rebalance(BaseModel):

    def __init__(self, opt):
        super(IRModel_TransWeather_Rebalance, self).__init__(opt)

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

        self.loss_snow = torch.Tensor([1e-1])[0].to(self.device)
        self.loss_derain = torch.Tensor([1e-1])[0].to(self.device)
        self.loss_dehaze = torch.Tensor([1e-1])[0].to(self.device)

        self.loss_snow_min = torch.Tensor([1e-1])[0].to(self.device)
        self.loss_derain_min = torch.Tensor([1e-1])[0].to(self.device)
        self.loss_dehaze_min = torch.Tensor([1e-1])[0].to(self.device)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        super().resume_training(resume_state)

        resume_loss_last = resume_state["loss_last"]
        resume_loss_min = resume_state["loss_min"]

        print("resuming from IRModel_Srresnet_yml resume_training")
        (
            self.loss_snow,
            self.loss_derain,
            self.loss_dehaze,
        ) = resume_loss_last
        (
            self.loss_snow_min,
            self.loss_derain_min,
            self.loss_dehaze_min,
        ) = resume_loss_min

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        self.ema_loss = train_opt.get("ema_loss", 0.01)
        self.tau = train_opt.get("tau", 0.1)

        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                params_ema = self.opt["path"].get("param_key_g_ema", "params_ema")
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    params_ema,
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
            self.cri_pixtype = Rebalance_L1().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=None):

        lq = data["lq"].to(self.device)
        lq = (lq - 0.5) / 0.5
        self.lq = lq

        if "gt" in data:
            self.gt = data["gt"].to(self.device)
        if "label" in data:
            self.typeword = data["label"]

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        loss_list = []
        loss_list_last = [
            self.loss_dehaze,
            self.loss_derain,
            self.loss_snow,
        ]
        loss_list_min = [self.loss_dehaze_min, self.loss_derain_min, self.loss_snow_min]

        if self.cri_pix:

            l_pix_DWA, loss_list, loss_grad_dict, weight_list_dict = self.cri_pixtype(
                restored=self.output,
                cleanpatch=self.gt,
                de_id=self.typeword,
                T=self.tau,
                loss_list_last=loss_list_last,
                loss_list_min=loss_list_min,
            )

            l_total += l_pix_DWA

            loss_dict["l_pix"] = l_total
            if current_iter % 1 == 0:
                a = self.ema_loss

                if loss_list[2] != 0:
                    self.loss_denoise_50 = (
                        a * loss_list[2].detach() + (1 - a) * self.loss_snow
                    )
                    if loss_list[2] < self.loss_snow_min:
                        self.loss_snow_min = loss_list[2].detach()

                if loss_list[1] != 0:
                    self.loss_derain = (
                        a * loss_list[1].detach() + (1 - a) * self.loss_derain
                    )
                    if loss_list[1] < self.loss_derain_min:
                        self.loss_derain_min = loss_list[1].detach()

                if loss_list[0] != 0:
                    self.loss_dehaze = (
                        a * loss_list[0].detach() + (1 - a) * self.loss_dehaze
                    )
                    if loss_list[0] < self.loss_dehaze_min:
                        self.loss_dehaze_min = loss_list[0].detach()

                loss_dict["loss_snow"] = self.loss_snow
                loss_dict["loss_derain"] = self.loss_derain
                loss_dict["loss_dehaze"] = self.loss_dehaze

                loss_dict["loss_snow_min"] = self.loss_snow_min
                loss_dict["loss_derain_min"] = self.loss_derain_min
                loss_dict["loss_dehaze_min"] = self.loss_dehaze_min

                loss_dict["loss_snow_grad"] = loss_grad_dict[2].detach()
                loss_dict["loss_derain_grad"] = loss_grad_dict[1].detach()
                loss_dict["loss_dehaze_grad"] = loss_grad_dict[0].detach()

                loss_dict["loss_snow_weight"] = weight_list_dict[2].detach()
                loss_dict["loss_derain_weight"] = weight_list_dict[1].detach()
                loss_dict["loss_dehaze_weight"] = weight_list_dict[0].detach()

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        self.loss_snow = torch.tensor(self.log_dict["loss_snow"]).to(self.device)
        self.loss_derain = torch.tensor(self.log_dict["loss_derain"]).to(self.device)
        self.loss_dehaze = torch.tensor(self.log_dict["loss_dehaze"]).to(self.device)

        self.loss_snow_min = torch.tensor(self.log_dict["loss_snow_min"]).to(
            self.device
        )
        self.loss_derain_min = torch.tensor(self.log_dict["loss_derain_min"]).to(
            self.device
        )
        self.loss_dehaze_min = torch.tensor(self.log_dict["loss_dehaze_min"]).to(
            self.device
        )

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        B, C, H, W = img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(img)
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[
            :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
        ]

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }
            self._initialize_best_metric_results(dataset_name)

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit="image")

        cnt = 0
        metric_data = dict()

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt["val"].get("grids", False):
                self.grids()

            self.test()

            if self.opt["val"].get("grids", False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr)
                del self.gt
            metric_data["img"] = sr_img
            metric_data["img2"] = gt_img

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(
                        self.opt["path"]["visualization"], dataset_name
                    )

                    imwrite(L_img, osp.join(visual_dir, f"{img_name}_L.png"))
                    imwrite(R_img, osp.join(visual_dir, f"{img_name}_R.png"))
                else:
                    if self.opt["is_train"]:

                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )

                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}_gt.png",
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f"{img_name}.png",
                        )
                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f"{img_name}_gt.png",
                        )

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt["val"]["metrics"])

                for name, opt_ in opt_metric.items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f"Test {img_name}")
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = (
                    torch.tensor(self.metric_results[metric]).float().to(self.device)
                )
            collected_metrics["cnt"] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt["rank"] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == "cnt":
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            if with_metrics:
                for metric in metrics_dict.keys():
                    self.metric_results[metric] = metrics_dict[metric]
                    # update the best metric result
                    self._update_best_metric_result(
                        dataset_name, metric, self.metric_results[metric], current_iter
                    )

            self._log_validation_metric_values(
                current_iter, dataloader.dataset.opt["name"], tb_logger
            )
        return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            # self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]])
                metric_data["img2"] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.png",
                    )
                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.png',
                        )
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f"Test {img_name}")
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train = True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
                "loss_last": [],
                "loss_min": [],
            }
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())

            state["loss_last"].extend(
                [self.loss_dehaze, self.loss_derain, self.loss_snow]
            )
            state["loss_min"].extend(
                [self.loss_dehaze_min, self.loss_derain_min, self.loss_snow_min]
            )

            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt["path"]["training_states"], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warn(
                        f"Save training state error: {e}, remaining retry times: {retry - 1}"
                    )
                    time.sleep(5)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:  # still cannot save
                raise IOError(f"Cannot save {save_path}.")
