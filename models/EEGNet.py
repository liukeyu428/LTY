import torch

from .networks import *

from torch.nn.utils import weight_norm




def DS_Combin_two(classes, alpha1, alpha2):
    # Calculate the merger of two DS evidences
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = classes / S[v]

    # b^0 @ b^(0+1)
    bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate K
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
    K = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
    # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

    # calculate new S
    S_a = classes / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return alpha_a



def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class CSNet(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 pool_mode='mean',
                 F1=16,
                 kernLength=64,
                 poolSize=7,
                 D=2,
                 dropout=0.1,
                 source_num=8,
                 ):
        super(CSNet, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self
        padding1 = (kernLength - 1) // 2
        padding2 = (16 - 1) // 2
        F2 = F1 * D
        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_chans, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), ceil_mode=False),
            nn.Dropout(dropout)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, padding2 + 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.AvgPool2d((1, poolSize)),
            nn.Dropout(dropout)
        )



        '''self.batch_conv = nn.ModuleList(nn.Sequential(
            TemporalConvNet(num_inputs=32, num_channels=[32, 32], kernel_size=2, dropout=0.3),
        )for i in range(self.source_num))'''

        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        #out = self.forward_init(out)
        # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.cls = nn.Sequential(
            Conv2dWithConstraint(32, self.n_classes,
                                 (1, 20), max_norm=0.5,
                                 bias=True),
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output)
        )
        self.output1 = nn.Softplus()
        self.apply(glorot_weight_zero_bias)
        self.weight = nn.Parameter(th.FloatTensor([1. / self.source_num] * self.source_num), requires_grad=True)
        #self.weight = nn.Parameter(th.FloatTensor([1.0] * self.source_num), requires_grad=True)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x


    def feature_extractor(self, x):
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.channel_conv(x)
        return x

    def forward_target_source(self, x):
        batch_size = x.shape[0]
        assert 0 == batch_size % (self.source_num + 1)
        batch_size_s = batch_size // (self.source_num + 1)# 每个源域的样本数量
        a = x[self.source_num * batch_size_s:]
        x = self.feature_extractor(x[self.source_num * batch_size_s:])
        #feed_together = x[self.source_num * batch_size_s:]
        output = self.spatial_conv(x)
        output = self.output1(self.cls(output))
        return output


    def forward_target_only(self, x):
        x = self.feature_extractor(x)
        target_feats = []
        target_cls = []
        processed_tensors = []
        output = self.spatial_conv(x)
        output = self.output1(self.cls(output))
        return target_feats, target_cls, output

    def forward_testother(self, x):
        batch_size = x.shape[0]
        assert 0 == batch_size % (self.source_num + 1)
        batch_size_s = batch_size // (self.source_num + 1)  # 每个源域的样本数量
        source_cls = []
        for i in range(self.source_num+1):
            x1 = x[i * batch_size_s:(i + 1) * batch_size_s]
            x1 = self.feature_extractor(x1)
            output = self.spatial_conv(x1)
            o = output[:batch_size_s]
            source_cls.append(self.output1(self.cls(output[:batch_size_s])))  # 每一个元素64,4的列表，8个元素，8各分支的源域分类结果
        all_cls = th.stack(source_cls, dim=1)
        return  source_cls



    '''def forward(self, x, is_target_only=True, ):
        if not is_target_only:
            return self.forward_target_source(x)
        return self.forward_target_only(x)'''


    def forward(self, x, is_target_only=True, is_best = False):
        if not is_best:
            if not is_target_only:
                return self.forward_target_source(x)
            return self.forward_target_only(x)
        return self.forward_testother(x)

class CEDL(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 source_num,
                 pool_mode='mean',
                 F1=16,
                 kernLength=64,
                 poolSize=7,
                 D=2,
                 dropout=0.1,
                 ):
        super(CEDL, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self
        padding1 = (kernLength - 1) // 2
        padding2 = (16 - 1) // 2
        F2 = F1 * D
        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_chans, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), ceil_mode=False),
            nn.Dropout(dropout)
        )

        self.spatial_conv =nn.ModuleList( nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, padding2 + 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.AvgPool2d((1, poolSize)),
            nn.Dropout(dropout)
        )for i in range(self.source_num))



        '''self.batch_conv = nn.ModuleList(nn.Sequential(
            TemporalConvNet(num_inputs=32, num_channels=[32, 32], kernel_size=2, dropout=0.3),
        )for i in range(self.source_num))'''

        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        #out = self.forward_init(out)
        # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.cls = nn.ModuleList(nn.Sequential(
            Conv2dWithConstraint(32, self.n_classes,
                                 (1, 20), max_norm=0.5,
                                 bias=True),
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output)
        ) for i in range(self.source_num))
        self.output1 = nn.Softplus()
        self.apply(glorot_weight_zero_bias)
        self.weight = nn.Parameter(th.FloatTensor([1. / self.source_num] * self.source_num), requires_grad=True)
        #self.weight = nn.Parameter(th.FloatTensor([1.0] * self.source_num), requires_grad=True)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x


    def feature_extractor(self, x):
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.channel_conv(x)
        return x

    def forward_target_source(self, x):
        batch_size = x.shape[0]
        assert 0 == batch_size % (self.source_num + 1)
        batch_size_s = batch_size // (self.source_num + 1)# 每个源域的样本数量
        source_cls = []
        target_cls = []
        target1_cls = []
        processed_tensors1 = []
        x = self.feature_extractor(x)
        for i in range(self.source_num):
            feed_together = th.cat([x[i * batch_size_s:(i + 1) * batch_size_s], x[self.source_num * batch_size_s:]])#选择第i个源域的样本和目标域拼接
            output = self.spatial_conv[i](feed_together)
            #print(self.cls[i](output[batch_size_s:]).shape)
            #source_cls.append(self.cls[i](output[:batch_size_s]))#每一个元素64,4的列表，8个元素，8各分支的源域分类结果
            #target_cls.append(self.cls[i](output[batch_size_s:]))#每一个64,4
            o = output[:batch_size_s]
            source_cls.append(self.output1(self.cls[i](output[:batch_size_s])))  # 每一个元素64,4的列表，8个元素，8各分支的源域分类结果
            target_cls.append (self.output1(self.cls[i](output[batch_size_s:]))) # 每一个元素64,4的列表，8个元素，8各分支的目标域域分类结果
            # source_feats.append(output[:batch_size_s])
            # target_feats.append(output[batch_size_s:])
            # source_feats.append(self.separable_conv[i](x[i * batch_size_s:(i + 1) * batch_size_s]))
            # target_feats.append(self.separable_conv[i](x[self.source_num * batch_size_s:]))
        # for i in range(self.source_num):
        #     source_cls.append(self.cls[i](source_feats[i]))
        #     target_cls.append(self.cls[i](target_feats[i]))
        #这两行代码的核心作用是：使用相同的分类器（self.cls[i]）分别处理源域和目标域的数据特征，并得到它们各自的分类结果。
        # 通过将源域和目标域数据传递给同一个分类器，模型可以保证源域和目标域在同一个特征空间下被处理，从而有助于跨域的特征对齐和分类。最终，这些分类结果会被分别存储在 source_cls 和 target_cls 列表中，用于进一步的计算或决策。
        all_cls = th.stack(target_cls, dim=1)#64，8，4的张量
        #cls = F.softmax(all_cls, dim=-1) * F.softmax(self.weight, dim=-1).view(1, -1, 1)#结果 cls 是一个形状为 [batch_size, num_sources, num_classes] 的张量，其中每个类别的概率已经根据不同源的权重进行了加权调整
        #cls = all_cls * self.weight.view(1, -1, 1)
        split_tensors = torch.split(all_cls, 1, dim=1)
        result_tensors = [t.squeeze(1) for t in split_tensors] #去掉维度1，八个64，1，4变为64，4
        for tensor in result_tensors:
            processed_tensor = tensor + 1
            processed_tensors1.append(processed_tensor)
        result = processed_tensors1[0]
        for processed_tensor in processed_tensors1[1:]:
            result = DS_Combin_two(self.n_classes, result, processed_tensor)
        #print("weight",weight)
        #cls = th.sum(all_cls, dim=1)#各分支加权计算后的分类结果，64，4的向量
        #cls = self.output1(cls)
        return source_cls, target_cls, result

    def forward_target_only(self, x):
        x = self.feature_extractor(x)
        target_feats = []
        target_cls = []
        processed_tensors = []
        for i in range(self.source_num):
            target_feats.append(self.spatial_conv[i](x))
        for i in range(self.source_num):
            output = target_feats[i]
            target_cls.append(self.output1(self.cls[i](output)))
            #target_cls.append(self.output1(self.cls[i](target_feats[i])))
        all_cls = th.stack(target_cls, dim=1)
        #cls = F.softmax(all_cls, dim=-1) * F.softmax(weight, dim=-1).view(1, -1, 1)
        #cls = all_cls * weight.view(1, -1, 1)
        split_tensors = torch.split(all_cls, 1, dim=1)
        result_tensors = [t.squeeze(1) for t in split_tensors]
        for tensor in result_tensors:
            processed_tensor = tensor + 1
            processed_tensors.append(processed_tensor)
        result = processed_tensors[0]
        for processed_tensor in processed_tensors[1:]:
            result = DS_Combin_two(self.n_classes,result, processed_tensor)
        #cls = th.sum(all_cls, dim=1)
        #cls = self.output1(cls)
        return target_feats, target_cls, result

    def forward(self, x, is_target_only=True):
        if not is_target_only:
            return self.forward_target_source(x)
        return self.forward_target_only(x)



class CEDLCross(BaseModel):
    def __init__(self,
                 in_chans,
                 input_time_length,
                 n_classes,
                 source_num,
                 pool_mode='mean',
                 F1=16,
                 kernLength=64,
                 poolSize=7,
                 D=2,
                 dropout=0.1,
                 ):
        super(CEDLCross, self).__init__()

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self
        padding1 = (kernLength - 1) // 2
        padding2 = (16 - 1) // 2
        F2 = F1 * D
        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_b_1_c_0),
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_chans, 1), bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), ceil_mode=False),
            nn.Dropout(dropout)
        )

        self.spatial_conv =nn.ModuleList( nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, padding2 + 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.AvgPool2d((1, poolSize)),
            nn.Dropout(dropout)
        )for i in range(self.source_num))


        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        #out = self.forward_init(out)
        # out = self.separable_conv(self.spatial_conv(self.temporal_conv(out)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.cls = nn.ModuleList(nn.Sequential(
            Conv2dWithConstraint(32, self.n_classes,
                                 (1, 20), max_norm=0.5,
                                 bias=True),
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output)
        ) for i in range(self.source_num))
        self.output1 = nn.Softplus()
        self.apply(glorot_weight_zero_bias)
        self.weight = nn.Parameter(th.FloatTensor([1. / self.source_num] * self.source_num), requires_grad=True)
        #self.weight = nn.Parameter(th.FloatTensor([1.0] * self.source_num), requires_grad=True)

    def forward_init(self, x):
        with th.no_grad():
            for module in self._modules:
                if isinstance(self._modules[module], th.nn.ModuleList):
                    x = self._modules[module][0](x)
                else:
                    x = self._modules[module](x)
        return x


    def feature_extractor(self, x):
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.channel_conv(x)
        return x

    def forward_target_source(self, x):
        batch_size = x.shape[0]
        assert 0 == batch_size % (self.source_num + 1)
        batch_size_s = batch_size // (self.source_num + 1)# 每个源域的样本数量
        source_cls = []
        target_cls = []
        target1_cls = []
        processed_tensors1 = []
        x = self.feature_extractor(x)
        for i in range(self.source_num):
            #feed_together = th.cat([x[i * batch_size_s:(i + 1) * batch_size_s], x[(i+1) * batch_size_s:(i + 2) * batch_size_s]])#选择第i个源域的样本和目标域拼接
            output = self.spatial_conv[i](x[i * batch_size_s:(i + 1) * batch_size_s])
            #print(self.cls[i](output[batch_size_s:]).shape)
            #source_cls.append(self.cls[i](output[:batch_size_s]))#每一个元素64,4的列表，8个元素，8各分支的源域分类结果
            #target_cls.append(self.cls[i](output[batch_size_s:]))#每一个64,4
            o = output[:batch_size_s]
            source_cls.append(self.output1(self.cls[i](output[:batch_size_s])))  # 每一个元素64,4的列表，8个元素，8各分支的源域分类结果
        return source_cls

    def forward_target_only(self, x, c=None):
        x = self.feature_extractor(x)
        target_feats = []
        target_cls = []
        processed_tensors = []
        for i in range(self.source_num):
            target_feats.append(self.spatial_conv[i](x))
        for i in range(self.source_num):
            output = target_feats[i]
            target_cls.append(self.output1(self.cls[i](output)))
            #target_cls.append(self.output1(self.cls[i](target_feats[i])))
        all_cls = th.stack(target_cls, dim=1)
        #cls = F.softmax(all_cls, dim=-1) * F.softmax(weight, dim=-1).view(1, -1, 1)
        #cls = all_cls * weight.view(1, -1, 1)
        split_tensors = torch.split(all_cls, 1, dim=1)
        result_tensors = [t.squeeze(1) for t in split_tensors]
        for tensor in result_tensors:
            processed_tensor = tensor + 1
            processed_tensors.append(processed_tensor)
        result = processed_tensors[0]
        for processed_tensor in processed_tensors[1:]:
            result = DS_Combin_two(self.n_classes,result, processed_tensor)
        #cls = th.sum(all_cls, dim=1)
        #cls = self.output1(cls)
        return target_feats, target_cls, result

    def forward(self, x, is_target_only=True):
        if not is_target_only:
            return self.forward_target_source(x)
        return self.forward_target_only(x)