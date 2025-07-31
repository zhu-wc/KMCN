import sys

from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention,MultiHeadAttentionWithBias
from ..relative_embedding import AllRelationalEmbedding

class GetViewCenter(nn.Module):
    def __init__(self, alpha=10.0):
        super(GetViewCenter, self).__init__()
        self.num_clusters = 18
        self.dim = 512
        self.alpha = alpha
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(1e-1 * torch.rand(self.num_clusters, self.dim)) # 生成指定形状的矩阵，其中的元素为0-1之间的随机数，与0.1相乘后即为0-0.1之间的随机数

        self._init_params()
    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, bias,mask):
        # x.shape = bs nor dim
        # bias.shape = bs h noA noB，期望得到bs h noA nok
        # mask.shape = bs 1 1 a
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [bs, nor, dim, 1] -> [bs, dim, nor, 1]

        x_change = x #[bs, dim, nor, 1]

        N, C = x_change.shape[:2] # bs dim
        soft_assign = self.conv(x_change).view(N, self.num_clusters, -1)  # [bs, nok, nor]
        soft_assign = F.softmax(soft_assign, dim=1) # 计算权重，形状为 bs nok noB

        x_flatten = x_change.reshape(N, C, -1)  # bs 512 nog

        # calculate residuals to each clusters   [300, num_cluster, 1024, M]
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

        # 换成一个attention
        output = vlad
        #output = torch.cat((vlad, x_flatten.transpose(1, 2)), dim=1)
        out_bias = torch.matmul(bias,soft_assign.permute(0,2,1).unsqueeze(1)) # bs h noA noB * bs 1 noB nok -> bs h noA nok
        cluster_mask = mask.data.new_full((N,1,1,self.num_clusters),False)
        # print(mask.shape,cluster_mask.shape)
        # print(mask.shape,cluster_mask.shape)
        cluster_mask = cluster_mask.repeat(1,1,mask.shape[2],1)
        out_mask = torch.cat([mask,cluster_mask],dim = -1)
        return output, out_bias,out_mask
class VisualSemanticComplementary(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(VisualSemanticComplementary, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)

        self.cluster = GetViewCenter()

        self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)
        self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, sa_bias,ca_bias,sa_attention_mask):
        '''
            queries: bs a dim
            keys: bs b dim
            sa_bias: bs h a a
            ca_bias的形状：bs h a a+b
        '''
        # SA运算
        att1 = self.mhatt1(queries, queries, queries, sa_bias,sa_attention_mask)
        att1 = self.lnorm1(queries + self.dropout1(att1))
        # ff1 = self.pwff1(att1)
        #估算视角中心（View Center）
        # pad_num = torch.sum(sa_attention_mask,dim=-1) #计算pad了多少个特征
        nf = queries.shape[1]
        # true_feature = queries[:,:-pad_num] # 仅取真实特征
        center,bias,ca_mask = self.cluster(keys,ca_bias[:,:,:,nf:],sa_attention_mask) # center.shape = bs K(9) 512 ;  bias.shape = bs h a K(9)
        # center = torch.sum(queries,dim=1,keepdim=True)/(nf-pad_num)#根据真实特征计算初始的 视角中心
        # center2 = self.mhatt2(center, queries, queries,sa_attention_mask[0]) #以MHA的方式对视角中心进行微调
        # center2 = self.lnorm2(center + self.dropout2(center2)) # bs 1 dim
        #利用视角中心拉近距离，然后执行CA运算
        # keys = (center2 + keys)*0.5
        all = torch.cat([queries,center],dim=1)
        ca_bias = torch.cat([sa_bias,bias],dim=-1) # sa_bias =
        att3 = self.mhatt3(queries,all,all,ca_bias,ca_mask)
        att3 = self.lnorm3(queries + self.dropout3(att3))

        ff = self.pwff3((att1+att3)*0.5)
        # ff3 = self.pwff3(att3)

        return ff

class VisualFeatureIntegration(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(VisualFeatureIntegration, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt1 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)

        self.mhatt2 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)

        self.mhatt3 = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.sa_object = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.sa_grid = MultiHeadAttentionWithBias(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff2 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)
        self.pwff3 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout4 = nn.Dropout(dropout)
        self.lnorm4 = nn.LayerNorm(d_model)
        self.pwff4 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.dropout5 = nn.Dropout(dropout)
        self.lnorm5 = nn.LayerNorm(d_model)
        self.pwff5 = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, object, grid, bias,attention_mask_object,attention_mask_grid, attention_weights=None):
        '''
            传入参数bias的形状为bs 8 99*99，并且99个特征的组成为[object,grid]
        '''
        bs,nog,dim = grid.shape
        main_object = object[:,:25]
        main_grid = grid.view(bs,7,7,dim)[:,1:6,1:6].reshape(bs,-1,dim) # bs 5*5 dim
        main_all = torch.cat([main_object,main_grid],dim=1) #bs 50 dim
        main_object_bias = torch.cat([bias[:,:,:25,:25],bias[:,:,:25,50:].view(bs,8,25,7,7)[:,:,:,1:6,1:6].reshape(bs,8,25,25)],dim=-1) # bs 8 25 25+25 [object2object,object2grid]
        main_grid_bias = torch.cat([bias[:,:,50:,:25].view(bs,8,7,7,25)[:,:,1:6,1:6,:].reshape(bs,8,25,25),bias[:,:,50:,50:].view(bs,8,7,7,7,7)[:,:,1:6,1:6,1:6,1:6].reshape(bs,8,25,25)],dim=-1) #bs 8 25 50
                                                                                                                                                    # [grid2object,grid2grid]
        main_all_bias = torch.cat([main_object_bias,main_grid_bias],dim=-2) # bs 8 50 50
        main_all_mask = torch.cat([attention_mask_object[:,:,:,:25],attention_mask_grid[:,:,:,:25]],dim=-1) # bs 1 1 50
        # 主元素之间的SA运算
        att1 = self.mhatt1(main_all, main_all, main_all, main_all_bias,main_all_mask)
        att1 = self.lnorm1(main_all + self.dropout1(att1))
        # Agent Attention运算恢复object和grid的维度
        att_object = self.mhatt2(object, main_object, att1[:,:25], bias[:,:,:50,:25], attention_mask_object[:,:,:,:25])
        att_object = self.lnorm2(object + self.dropout2(att_object))
        # att_object = self.pwff2(att_object)
        att_grid = self.mhatt3(grid, main_grid, att1[:, 25:], bias[:,:,50:,50:].view(bs,8,49,7,7)[:,:,:,1:6,1:6].reshape(bs,8,49,25), attention_mask_grid[:,:,:,:25])
        att_grid = self.lnorm3(grid + self.dropout3(att_grid))
        # att_grid = self.pwff3(att_grid)
        #残差连接并丰富特征多样性
        sa_object = self.sa_object(object,object,object,bias[:,:,:50,:50],attention_mask_object)
        sa_object = self.lnorm4(object + self.dropout4(sa_object))
        ff1 = self.pwff4((sa_object+att_object)*0.5)

        sa_grid = self.sa_grid(grid,grid,grid,bias[:,:,50:,50:],attention_mask_grid)
        sa_grid = self.lnorm5(grid + self.dropout5(sa_grid))
        ff2 = self.pwff5((sa_grid+att_grid)*0.5)

        return ff1,ff2

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.object2ctx = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.grid2ctx =nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.ctx2grid = nn.ModuleList([VisualSemanticComplementary(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.vfie = nn.ModuleList([VisualFeatureIntegration(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, object,grid,txt_ctx,global_visual,bound_box, attention_weights=None):
        '''
        object: bs nor 512
        grid: bs nog 512
        txt_ctx: bs 48 512
        global_visual: bs 512
        bound_box: bs nor 6
        '''
        # print(object.shape,grid.shape,txt_ctx.shape)
        assert object.shape[1]==50
        assert grid.shape[1] == 49
        assert txt_ctx.shape[1] == 48
        n_object, n_grid, n_ctx = object.shape[1], grid.shape[1], txt_ctx.shape[1]
        object_mask = (torch.sum(torch.abs(object), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        grid_mask = (torch.sum(torch.abs(grid), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        txt_ctx_mask = (torch.sum(torch.abs(txt_ctx), -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        #计算bias
        # 1. 构建bs 49 49 64
        relative_geometry_embeddings = AllRelationalEmbedding(bound_box) #bs 147 147 64
        # 2. 引入可学习参数linear 64->1
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        # 3. cat拼接 以及 relu激活
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights) # bs 8 147 147

        # 4. 为模块切分传入参数
        object2ctx = torch.cat([relative_geometry_weights[:,:,:50,:50],relative_geometry_weights[:,:,:50,99:]],dim=-1) # object->object+ctx
        grid2ctx = relative_geometry_weights[:,:,50:99,50:] #grid->grid+ctx
        ctx2grid = torch.cat([relative_geometry_weights[:,:,99:,99:],relative_geometry_weights[:,:,99:,50:99]],dim=-1)#ctx->ctx+grid

        # out = torch.cat([object,grid,txt_ctx],dim=1) # bs 147 512
        out_object,out_grid,out_ctx = object,grid,txt_ctx

        tmp_mask1 = torch.eye(object.shape[1], device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1)# bs * 1 * nor * nor
        # tmp_mask2 = torch.ones(object.shape[1],txt_ctx.shape[1],device=object.device).unsqueeze(0).unsqueeze(0).repeat(object.shape[0],1,1,1) # bs 1 nor noc
        object_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * nor *(nor+noc)

        tmp_mask1 = torch.eye(grid.shape[1], device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1)# bs * 1 * nog * nog
        # tmp_mask2 = torch.ones(grid.shape[1],txt_ctx.shape[1],device=grid.device).unsqueeze(0).unsqueeze(0).repeat(grid.shape[0],1,1,1) # bs 1 nog noc
        grid_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * nor *(nor+noc)

        tmp_mask1 = torch.eye(txt_ctx.shape[1], device=txt_ctx.device).unsqueeze(0).unsqueeze(0).repeat(txt_ctx.shape[0],1,1,1)# bs * 1 * noc * noc
        # tmp_mask2 = torch.ones(txt_ctx.shape[1],grid.shape[1],device=txt_ctx.device).unsqueeze(0).unsqueeze(0).repeat(txt_ctx.shape[0],1,1,1) # bs 1 noc nog
        ctx_mask2 = (tmp_mask1 == 0) # (torch.cat([tmp_mask1, tmp_mask2], dim=-1) == 0) # bs * 1 * noc *(noc+nog)


        for o2c,g2c,c2g,vfie in zip(self.object2ctx,self.grid2ctx,self.ctx2grid,self.vfie):
            temp_object = o2c(out_object,out_ctx,relative_geometry_weights[:,:,:50,:50],object2ctx,object_mask2)
            temp_grid = g2c(out_grid,out_ctx,relative_geometry_weights[:,:,50:99,50:99],grid2ctx,grid_mask2)
            out_ctx = c2g(out_ctx,out_grid,relative_geometry_weights[:,:,99:,99:],ctx2grid,ctx_mask2)

            out_object,out_grid = vfie(temp_object,temp_grid,relative_geometry_weights[:,:,:99,:99],object_mask,grid_mask)

        out = torch.cat([out_object,out_grid,out_ctx],dim=1)
        attention_mask = torch.cat([object_mask, grid_mask, txt_ctx_mask], dim=-1)
        return out,attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)

    def forward(self, obj,grid, txt_ctx, global_visual,bound_box, attention_weights=None):
        return super(TransformerEncoder, self).forward(obj,grid,txt_ctx,global_visual,bound_box, attention_weights=attention_weights)
