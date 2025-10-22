# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = ['Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder']


class Detect(nn.Module):
    """用于YOLOv8模型的检测头部。YOLOv8 Detect head for detection models."""
    dynamic = False             # 该属性用于控制是否进行动态网格重建。如果设为True，则会启用动态网格重建，否则将按照固定的方式处理网络结构。force grid reconstruction
    export = False              # 此属性表示是否处于导出模式。当设置为True时，可能会启用某种导出模式，在这种模式下执行特定的操作或输出特定的结果。export mode
    shape = None                # 这个属性用于存储形状信息。在模型运行过程中，可能需要在不同的步骤中记录输入数据的形状等信息，以便后续处理。
    anchors = torch.empty(0)    # 这是一个用于存储锚点信息的张量。在目标检测任务中，锚点通常用于生成预测边界框。init
    strides = torch.empty(0)    # 这个属性用于存储步幅信息。在卷积神经网络中，步幅决定了卷积核在输入上滑动的距离，影响特征图的尺寸。init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc                        # 类别数，表示待检测的对象类别数量，默认为80。number of classes
        self.nl = len(ch)                   # ch：是一个元组，代表每个检测层的输出通道数。nl:记录检测层数量，即有几个检测层。number of detection layers
        self.reg_max = 16                   # 设置DFL通道数上限，用于调节特征图的尺寸变化。DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4     # 计算每个锚点的输出数，包括类别预测和定位信息。number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # 初始化用于存储步幅信息的张量，长度为检测层数量，用于在构建过程中计算。strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # 计算通道数，确保合适的特征提取和输出处理。channels
        self.cv2 = nn.ModuleList(           # 分别是针对每个检测层的卷积操作序列，用于处理定位信息和类别信息，通过nn.ModuleList组织起来。
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()     # 如果DFL通道数大于1，则创建DFL模块用于特征学习，否则使用nn.Identity()作为占位符模块。

    def forward(self, x):
        """
        前向传播方法，主要用于处理特征张量并生成预测的边界框和类别概率。
        Concatenates and returns predicted bounding boxes and class probabilities.
        """
        shape = x[0].shape                          # 获取输入数据的形状，假设为BCHW。
        for i in range(self.nl):                    # 循环遍历所有检测层：
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)   # 对每个检测层，将经过两个卷积操作融合后的特征张量进行拼接，即将定位信息和类别信息合并在一起。最终得到处理后的特征张量列表x。
        if self.training:                           # 如果处于训练阶段（self.training == True）：
            return x                                # 直接返回处理后的特征张量列表x
        elif self.dynamic or self.shape != shape:   # 若为动态模式或输入形状发生变化（self.dynamic == True 或 self.shape != shape）：
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)) # 通过调用make_anchors函数，根据当前特征张量和步幅信息计算锚点信息，并更新self.anchors和self.strides。
            self.shape = shape                      # 更新记录的形状信息self.shape。

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 将处理后的特征张量进行重塑和连接：针对每个检测层的特征张量，根据形状进行视图重塑，并在最后一个维度上连接起来，得到x_cat。
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # 根据导出模式和格式，分割出定位信息和类别信息：avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]       # 如果处于导出模式且格式符合一定条件，则直接将x_cat划分为定位信息和类别信息
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 否则，使用切分函数split，按指定维度（1维）将x_cat分割为定位信息和类别信息。

        # 进行边界框的转换和处理：
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides # 将定位信息应用于预测的边界框，并根据锚点信息和步幅信息计算实际坐标。
        y = torch.cat((dbox, cls.sigmoid()), 1)     # 结合类别概率进行sigmoid处理，得到最终的输出结果y。
        return y if self.export else (y, x)         # 如果处于导出模式，则返回处理后的结果y；否则返回元组(y, x)，包含处理后的结果以及特征张量列表x。

    def bias_init(self):
        """
        用于初始化检测头部的偏置项。
        帮助模型更好地适应训练数据并提高性能。通过这些初始化操作，可以为模型提供合适的初始条件，有助于加快收敛速度并提高准确性。
        Initialize Detect() biases, WARNING: requires stride availability.
        """
        m = self  # 将当前实例赋值给变量m，以便在后续操作中引用。self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 对cv2和cv3这两个卷积层序列进行遍历，同时也使用了stride信息。from
            a[-1].bias.data[:] = 1.0  # 设置定位信息（box）的偏置项为1.0，这可能有助于模型更好地学习目标的位置信息。box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 设置类别信息（cls）的偏置项，cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """
    用于分割模型的YOLOv8 Segment头部。
    用于分割模型中的头部设计，负责生成掩码原型并对输入进行相应处理。通过这个模块，可以实现对图像的语义分割任务。
    YOLOv8 Segment head for segmentation models.
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)                    # 继承自Detect类，传递了类别数nc和输出通道数ch。
        ###### Jiayuan changed self.nm to self.nc
        self.npr = 32                               # 中间卷积特征维度npr。intermediate convolutional feature dimension
        self.cv1 = Conv(ch[0], self.npr, k=3)       # 第一个卷积层cv1
        self.upsample = nn.ConvTranspose2d(self.npr, self.npr//2, 2, 2, 0, bias=True)  # 上采样层upsample。nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(self.npr//2, self.npr//4, k=3)  # 后续卷积层cv2、cv3
        self.cv3 = Conv(self.npr//4, self.nc+1)     ###### self.nc+1 means add the background
        self.sigmoid = nn.Sigmoid()                 # Sigmoid激活函数sigmoid
        # self.detect = Detect.forward
        #
        # c4 = max(ch[0] // 4, self.nm)
        # self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.cv3(self.cv2(self.upsample(self.cv1(x[0]))))   # 输入数据经过一系列卷积和上采样操作产生掩码原型p。mask protos
        if self.training:                                       # 如果处于训练阶段
            return p                                            # 则直接返回掩码原型信息
        return p                                                # 否则，返回掩码原型信息。
        # bs = p.shape[0]  # batch size
        # 处理了关于掩码系数的计算等内容，原始计划可能包括更多的处理操作，如额外的卷积处理和特征融合等。
        # mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        # x = self.detect(self, x)
        # if self.training:
        #     return x, mc, p
        # return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 1:2].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hidden_dim=256,
            num_queries=300,
            strides=(8, 16, 32),  # TODO
            nl=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        assert len(ch) <= nl
        assert len(strides) == len(ch)
        for _ in range(nl - len(strides)):
            strides.append(strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = strides
        self.nl = nl
        self.nc = nc
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # backbone feature projection
        self._build_input_proj_layer(ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, act, nl,
                                                          num_decoder_points)
        self.decoder = DeformableTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, nc) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def forward(self, feats, gt_meta=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            raise NotImplementedError
            # denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
            #     get_contrastive_denoising_training_group(gt_meta,
            #                                 self.num_classes,
            #                                 self.num_queries,
            #                                 self.denoising_class_embed.weight,
            #                                 self.num_denoising,
            #                                 self.label_noise_ratio,
            #                                 self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask = None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        if not self.training:
            out_logits = out_logits.sigmoid_()
        return out_bboxes, out_logits  # enc_topk_bboxes, enc_topk_logits, dn_meta

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    def _build_input_proj_layer(self, ch):
        self.input_proj = nn.ModuleList()
        for in_channels in ch:
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
        in_channels = ch[-1]
        for _ in range(self.nl - len(ch)):
            self.input_proj.append(
                nn.Sequential(nn.Conv2D(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
            in_channels = self.hidden_dim

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=torch.float32),
                                            torch.arange(end=w, dtype=torch.float32),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([h, w]).to(torch.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors.to(device=device, dtype=dtype), valid_mask.to(device=device)

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.nl > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.nl):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        memory = torch.where(valid_mask, memory, 0)
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)  # (bs, h*w, nc)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # (bs, h*w, 4)

        # (bs, topk)
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # extract region proposal boxes
        # (bs, topk_ind)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        topk_ind = topk_ind.view(-1)

        # Unsigmoided
        reference_points_unact = enc_outputs_coord_unact[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = enc_outputs_class[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
