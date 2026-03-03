from torch_sparse import SparseTensor
from Model.SAP import SelfAttentionPooling
from torch_geometric.nn import TransformerConv, GCNConv
from torch.nn import GroupNorm, LayerNorm
import dgl.function as fn
import numpy as np
import torch
from dgl.nn.functional import edge_softmax

from CASTLE.Model.function import get_activation_func
import torch.nn.functional as F

#########################################替换图卷积########################################
#################################### GTN #####################################
class Res_GTN(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Res_GTN, self).__init__()
        self.conv1 = TransformerConv(num_feature, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv2 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv3 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)

        self.gn = GroupNorm(15, out_dim)
        self.lin4 = torch.nn.Linear(out_dim, out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.gelu(x)
        return x

class Res_GCN(torch.nn.Module):
    def __init__(self, num_feature, out_dim, dropout_rate=0.5):
        super(Res_GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.conv3 = GCNConv(out_dim, out_dim)

        # self.gn = GroupNorm(15, out_dim)
        self.ln = LayerNorm(out_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = self.ln(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.gn(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.ln(x)
        x = F.gelu(x)
        return x
##################################################################################

from torch import nn
import torch
from einops import rearrange

class Interaction_GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_GCN, self).__init__()
        self.fc = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, inputs):
        x = inputs.mean(dim=1, keepdim=True)
        return self.fc(x)


class Interaction_SAGE(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_SAGE, self).__init__()
        self.fc_l = nn.Linear(hidden_channels, hidden_channels)
        self.fc_r = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, inputs):
        neighbor = inputs.mean(dim=1, keepdim=True)
        neighbor = self.fc_r(neighbor)
        x = self.fc_l(inputs)
        x = (x + neighbor)
        return x


class Interaction_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class N2N(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hop, dropout, activation,
                 feature_inter, inter_layer, feature_fusion, norm_type):
        super().__init__()
        self.num_hop = num_hop
        self.feature_inter_type = feature_inter
        self.feature_fusion = feature_fusion
        self.dropout = nn.Dropout(dropout)
        self.pre = False
        self.norm_type = norm_type
        self.build_activation(activation)

        # encoder
        self.fc = nn.Linear(in_channels, hidden_channels)

        # hop_embedding
        self.hop_embedding = nn.Parameter(torch.randn(1, num_hop, hidden_channels))
        # interaction
        self.build_feature_inter_layer(feature_inter, hidden_channels, inter_layer)

        # fusion
        if self.feature_fusion == 'attention':
            self.atten_self = nn.Linear(hidden_channels, 1)
            self.atten_neighbor = nn.Linear(hidden_channels, 1)
        # norm
        self.build_norm_layer(hidden_channels, inter_layer * 2 + 2)
        print('N2N hidden:', hidden_channels, 'interaction:', feature_inter, 'hop:', num_hop, 'layers:', inter_layer)

    def build_activation(self, activation):
        if activation == 'tanh':
            self.activate = F.tanh
        elif activation == 'sigmoid':
            self.activate = F.sigmoid
        elif activation == 'gelu':
            self.activate = F.gelu
        else:
            self.activate = F.relu

    def preprocess(self, adj, x):
        h0 = []
        for i in range(self.num_hop):
            h0.append(x)
            x = adj.to_dense().matmul(x)
        self.h0 = torch.stack(h0, dim=1)
        # self.pre = True
        return self.h0

    def build_feature_inter_layer(self, feature_inter, hidden_channels, inter_layer):
        self.interaction_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        if feature_inter == 'mlp':
            for i in range(inter_layer):
                mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
                self.interaction_layers.append(mlp)
        elif feature_inter == 'gcn':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_GCN(hidden_channels))
        elif feature_inter == 'sage':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_SAGE(hidden_channels))
        elif feature_inter == 'attention':
            for i in range(inter_layer):
                self.interaction_layers.append(
                    Interaction_Attention(hidden_channels, heads=4, dropout=0.))
        else:
            self.interaction_layers.append(torch.nn.Identity())

    def build_norm_layer(self, hidden_channels, layers):
        self.norm_layers = nn.ModuleList()
        for i in range(layers):
            if self.norm_type == 'bn':
                self.norm_layers.append(nn.BatchNorm1d(self.num_hop))
            elif self.norm_type == 'ln':
                self.norm_layers.append(nn.LayerNorm(hidden_channels))
            else:
                self.norm_layers.append(nn.Identity())

    def norm(self, h, layer_index):
        h = self.norm_layers[layer_index](h)
        return h

    # N * hop * d => N * hop * d
    def embedding(self, h):
        h = self.dropout(h)
        h = self.fc(h)
        h = h + self.hop_embedding
        h = self.norm(h, 0)
        return h

    # N * hop * d =>  N * hop * d
    def interaction(self, h):
        inter_layers = len(self.interaction_layers)
        for i in range(inter_layers):
            h_prev = h
            h = self.dropout(h)
            h = self.interaction_layers[i](h)
            h = self.activate(h)
            h = h + h_prev
            h = self.norm(h, i + 1)
        return h

    # N * hop * d =>  N * hop * d (concat) or N * d (mean/max/attention)
    def fusion(self, h):
        h = self.dropout(h)
        if self.feature_fusion == 'max':
            h = h.max(dim=1).values
        elif self.feature_fusion == 'attention':
            h_self, h_neighbor = h[:, 0, :], h[:, 1:, :]
            h_self_atten = self.atten_self(h_self).view(-1, 1)
            h_neighbor_atten = self.atten_neighbor(h_neighbor).squeeze()
            h_atten = torch.softmax(F.leaky_relu(h_self_atten + h_neighbor_atten), dim=1)
            h_neighbor = torch.einsum('nhd, nh -> nd', h_neighbor, h_atten).squeeze()
            h = h_self + h_neighbor
        else:  # mean
            h = h.mean(dim=1)
        h = self.norm(h, -1)
        return h

    def build_hop(self, adj, inputs):
        if len(inputs.shape) == 3:
            h = inputs
        else:
            if self.pre == False:
                self.h0 = self.preprocess(adj, inputs)
            h = self.h0
        return h

    def forward(self, g, inputs, devices):
        # step-1 the first preprocess of hop-information for accerelate training
        adj = g.adjacency_matrix().indices().to(torch.int64)
        v = torch.ones(adj.shape[1]).to(devices)
        sparse_adj = torch.sparse.FloatTensor(adj, v, torch.Size([g.num_nodes(), g.num_nodes()]))
        h = self.build_hop(sparse_adj, inputs)
        # step-2 hop-embedding
        h = self.embedding(h)
        # step-3 hop-interaction
        h = self.interaction(h)
        # step-4 hop-fusion
        h = self.fusion(h)
        return h



class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='ReLU', bias=True):
        super(DenseLayer, self).__init__()
        if activation is not None:
            self.act = get_activation_func(activation)
        else:
            self.act = None
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_feat):
        if self.act is not None:
            return self.act(self.fc(input_feat))
        else:
            return self.fc(input_feat)


def distance(edge_feat):
    def func(edges):
        return {'dist': (edges.src[edge_feat] - edges.dst[edge_feat]).pow(2).sum(dim=-1).sqrt()}
    return func


def edge_cat(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat([edges.data[src_field], edges.data[dst_field]], dim=-1)}
    return func


class AtomPooling(nn.Module):
    def __init__(self, input_dim):
        super(AtomPooling, self).__init__()
        self.pool = SelfAttentionPooling(input_dim, input_dim)

    def forward(self, atom_features, index_list):
        x = []
        index = index_list.tolist()

        for st2end in index:
            x.append(self.pool(atom_features[int(st2end[0]):int(st2end[1]) + 1, :]))
        # for i in range(len(index)-1):
        #     x.append(self.pool(atom_features[index[i]:index[i+1], :]))
        y = torch.cat((x[0], x[1]), dim=0)
        for j in range(2, len(x)):
            y = torch.cat((y, x[j]), dim=0)
        return y



class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_dim, act="ReLU"):
        super(ResidualLayer, self).__init__()
        self.act = get_activation_func(act)
        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation=act)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=act)

    def forward(self, he):
        return he + self.lin2(self.lin1(he))


class Distance2embedding(nn.Module):

    def __init__(self, hidden_dim, cut_dist, activation="ReLU"):
        super(Distance2embedding, self).__init__()

        self.cut_dist = cut_dist

        self.dist_embedding_layer = nn.Embedding(int(cut_dist) - 1, hidden_dim)

        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)

    def forward(self, dist_feat):
        dist = torch.clamp(dist_feat.squeeze(), 1.0, self.cut_dist - 1e-6).type(
            torch.int64) - 1

        distance_emb = self.dist_embedding_layer(dist)

        distance_emb = self.dist_input_layer(distance_emb)

        return distance_emb


class Angle2embedding(nn.Module):

    def __init__(self, hidden_dim, class_num, activation="ReLU"):
        super(Angle2embedding, self).__init__()

        self.class_num = class_num
        self.angle_embedding_layer = nn.Embedding(int(class_num), hidden_dim)
        self.angle_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)

    def forward(self, angle):
        angle = (angle / (3.1415926 / self.class_num)).type(torch.int64)  # 对角度分成六个区间，0~30,30~60,60~90,......
        angle_emb = self.angle_embedding_layer(angle)  # 对角度进行编码，编码成节点维度的向量
        angle_emb = self.angle_input_layer(angle_emb)

        return angle_emb


class pos2embedding(nn.Module):

    def __init__(self, hidden_dim, pos_coordinate, activation="ReLU"):
        super(pos2embedding, self).__init__()

        self.pos_coordinate = pos_coordinate
        self.pos_embedding_layer = DenseLayer(int(pos_coordinate), hidden_dim)
        self.pos_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)

    def forward(self, pos):
        pos_emb = self.pos_embedding_layer(pos)  # 对角度进行编码，编码成节点维度的向量
        pos_emb = self.pos_input_layer(pos_emb)

        return pos_emb


class dis2embedding(nn.Module):

    def __init__(self, hidden_dim, dis_dim, activation="ReLU"):
        super(dis2embedding, self).__init__()

        self.dis_embedding_layer = DenseLayer(int(dis_dim), hidden_dim)
        self.dis_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)

    def forward(self, dis):
        dis_emb = self.dis_embedding_layer(dis)  # 对角度进行编码，编码成节点维度的向量
        dis_emb = self.dis_input_layer(dis_emb)

        return dis_emb


def src_cat_edge(src_field, edge_field, out_field):  # 将边的源节点和边的特征拼接，再转换维度成节点的特征维度
    def func(edges):
        x = torch.cat((edges.src[src_field], edges.data[edge_field]), dim=-1)
        return {out_field: x}

    return func


def norm_attn(attn_tensor):
    # 找到张量中的最大值和最小值
    min_val = attn_tensor.min()
    max_val = attn_tensor.max()

    # 对张量进行归一化
    normalized_tensor = (attn_tensor - min_val) / (max_val - min_val) - 1
    return normalized_tensor


class Atom2BondLayer_1(nn.Module):
    """
    Initial hidden representations for edges.
    """

    def __init__(self, node_dim, edge_dim, activation="ReLU"):  # 隐藏层的维度为200
        super(Atom2BondLayer_1, self).__init__()
        self.lin1 = DenseLayer(node_dim + edge_dim, node_dim, activation=activation, bias=True)  # 维度转换再加个激活函数

    def forward(self, g, atom_embedding, edge_embedding):
        with g.local_scope():
            g.ndata['h'] = atom_embedding
            g.edata['h'] = edge_embedding
            g.apply_edges(src_cat_edge('h', 'h', 'h'))
            h = self.lin1(g.edata['h'])
        return h


class Atom2BondLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, feat_drop, activation="ReLU"):  # 隐藏层的维度为200
        super(Atom2BondLayer, self).__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.lin1 = nn.Linear(edge_dim, node_dim)  # 维度转换再加个激活函数
        self.lnorm_lin1 = nn.LayerNorm(node_dim)
        self.atten_mid = nn.Linear(node_dim, 1)
        self.lnorm_e = nn.LayerNorm(node_dim)
        self.act = get_activation_func(activation)
        self.lin_beta = nn.Linear(node_dim * 3, 1, bias=False)

    def src_avg_edge(self, src_field, edge_field, out_field):  # 将边的源节点和边的特征拼接，再转换维度成节点的特征维度
        def func(edges):
            mid_cat = torch.stack((edges.src[src_field], edges.data[edge_field]), dim=1)
            mid_attn = self.atten_mid(mid_cat).squeeze()
            h_atten = torch.softmax(F.leaky_relu(mid_attn), dim=1)
            h_attn_fea = torch.einsum('nhd, nh -> nd', mid_cat, h_atten).squeeze()

            beta = self.lin_beta(
                torch.cat([h_attn_fea, edges.src[src_field], h_attn_fea - edges.src[src_field]], dim=-1))
            beta = beta.sigmoid()
            h = beta * edges.src[src_field] + (1 - beta) * h_attn_fea
            return {out_field: h}

        return func

    def forward(self, g, atom_embedding, edge_embedding):
        with g.local_scope():
            atom_fea = self.feat_drop(atom_embedding)
            edge_fea = self.feat_drop(edge_embedding)
            g.ndata['h'] = atom_fea
            g.edata['h'] = self.lnorm_lin1(self.lin1(edge_fea))
            g.apply_edges(self.src_avg_edge('h', 'h', 'h'))
            return self.act(self.lnorm_e(g.edata['h']))


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.lnorm_muti = nn.LayerNorm(d_model)

    def forward(self, inputs, devices):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(self.lnorm_muti(inputs))
        return nn.LayerNorm(self.d_model).to(devices)(output + residual)


class Bond2BondLayer(nn.Module):

    def __init__(self, edge_dim, num_head, feat_drop, attn_drop, activation="ReLU", activation_att="LeakyReLU",
                 class_num=6, beta=False):
        super(Bond2BondLayer, self).__init__()

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(edge_dim / num_head)
        self.num_head = num_head
        self.hidden_dim = edge_dim
        self.beta = beta
        if self.beta:
            self.lin_beta = nn.Linear(3 * edge_dim, 1, bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)

        self.k = nn.Linear(edge_dim, edge_dim, bias=False)
        self.q = nn.Linear(edge_dim, edge_dim, bias=False)
        self.v = nn.Linear(edge_dim, edge_dim, bias=False)
        self.dis_lin = nn.Linear(1, num_head, bias=False)
        self.act_fea = get_activation_func(activation)

        self.ffn = PoswiseFeedForwardNet(edge_dim, 2 * edge_dim)
        self.lnorm_muti = nn.LayerNorm(edge_dim)

        # angle information
        self.angle_embedding = Angle2embedding(edge_dim, class_num)
        self.angle1 = DenseLayer(edge_dim, edge_dim)
        self.angle2 = DenseLayer(edge_dim, edge_dim)

    def forward(self, graph, bond_embedding, index_kj, index_ji, idx_i, idx_j, idx_k, devices):
        with graph.local_scope():
            pos = graph.ndata["node_coordinate"]

            # Calculate angles.
            pos_i = pos[idx_i]
            pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i  # 俩个原子的向量相减得到边的向量
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki).norm(dim=-1)  # b向量垂直于 pos_ji和pos_ki组成的平面
            angle = torch.atan2(b, a)  # [0,3.14]  算出来的是k,j,i三个原子之间的夹角

            angle_embedding = self.angle_embedding(angle)
            angle_embedding = self.angle1(self.angle2(angle_embedding)).view(-1, self.num_head, self.out_feat)

            dist_decay = graph.edata["basic_attn"][index_kj].unsqueeze(dim=1)
            dist_decay = self.dis_lin(dist_decay).permute(0, 2, 1)
            bond_embedding_feats = self.feat_drop(bond_embedding)

            feat_kj = self.k(bond_embedding_feats).view(-1, self.num_head, self.out_feat)[index_kj]
            feat_ji = self.q(bond_embedding_feats).view(-1, self.num_head, self.out_feat)[index_ji]
            feat_kj_v = self.v(bond_embedding_feats).view(-1, self.num_head, self.out_feat)[index_kj] + angle_embedding

            feat = torch.mul(feat_kj + angle_embedding, feat_ji + angle_embedding)
            att = feat.sum(dim=-1, keepdim=True)
            sqrt_d = torch.sqrt(torch.tensor([self.out_feat]).to(devices))
            att = att / sqrt_d
            att_decay = att + dist_decay

            # soft max
            att_decay = torch.exp(att_decay)
            att_all = torch.zeros(len(bond_embedding), self.num_head, 1).to(bond_embedding.device)
            att_all = att_all.index_add_(0, index_ji, att_decay)
            att_all = att_all[index_ji]
            att_decay = self.attn_drop(att_decay / att_all)

            # ffn
            v_att = (feat_kj_v * att_decay).view(-1, self.hidden_dim)  # 这里计算的就是KJ与注意力相乘
            v_clone = bond_embedding.clone()
            v_clone = v_clone.index_add_(0, index_ji,
                                         v_att) - bond_embedding  # 将v_att中的元素加到v_clone的index_ji指定位置上，然后从结果上减去最原始的v_clone，并将结果保存在v_clone中
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([v_clone, bond_embedding, v_clone - bond_embedding], dim=-1))
                beta = beta.sigmoid()
                he = beta * bond_embedding + (1 - beta) * v_clone
            else:
                he = v_clone + bond_embedding
            he = self.ffn(he, devices)
            return he


def e_mul_e(edge_field1, edge_field2, out_field):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: edges.data[edge_field1] * edges.data[edge_field2]}

    return func


def u_cat_v(edges):
    u = edges.src['k']
    v = edges.dst['q']
    result = torch.cat((u, v), dim=-1)
    return {'e': result}


class Bond2AtomLayer(nn.Module):

    def __init__(self, node_dim, num_head, feat_drop, attn_drop, activation="ReLU",
                 activation_att="LeakyReLU", dist_cutoff=5, beta=True):
        super(Bond2AtomLayer, self).__init__()

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(node_dim / num_head)
        self.num_head = num_head
        self.node_dim = node_dim
        # self.edge_dim = edge_dim
        self.beta = beta

        self.k = nn.Linear(node_dim, node_dim, bias=False)
        self.q = nn.Linear(node_dim, node_dim, bias=False)
        self.v = nn.Linear(node_dim, node_dim, bias=False)
        self.act_fea = get_activation_func(activation_att)
        self.dis_lin = nn.Linear(1, num_head, bias=False)
        self.dis_lnorm = nn.LayerNorm(num_head)

        self.ffn = PoswiseFeedForwardNet(node_dim, 2 * node_dim)
        self.lnorm_muti = nn.LayerNorm(node_dim)

        if self.beta:
            self.lin_beta = nn.Linear(3 * node_dim, 1, bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)

    def adjust_node_positions(self, nx_g, center_node):
        g_pos = {}  # 存储节点的位置信息
        center_pos = (0, 0)  # 指定节点的位置为原点
        g_pos[center_node] = center_pos  # 将指定节点的位置设为原点

        num_nodes = nx_g.number_of_nodes() - 1  # 不包括指定节点
        radius = 1.5  # 围绕指定节点的半径
        angle_step = 2 * np.pi / num_nodes  # 计算每个节点的角度步长

        angle_offset = np.pi / 2  # 设置偏移角度，让指定节点位于中心
        angle = angle_offset  # 初始化角度
        for node in nx_g.nodes():
            if node == center_node:  # 跳过指定节点
                continue
            # 根据角度计算节点的位置
            distance = np.random.uniform(1, 1.5)  # 随机生成节点到指定节点的距离
            x = center_pos[0] + distance * radius * np.cos(angle)
            y = center_pos[1] + distance * radius * np.sin(angle)
            g_pos[node] = (x, y)  # 更新节点的位置
            # 更新角度
            angle += angle_step + np.random.uniform(-np.pi / 90, np.pi / 90)  # 加入随机性，使得不太均匀

        return g_pos

    def dot_product_message(self, edges):
        # 计算源节点和目标节点特征的点乘
        return {'message': torch.sum(edges.src['h'] * edges.dst['h'], dim=1)}

    def forward(self, graph, bond_embedding, node_embedding, devices, att_para=False):
        with graph.local_scope():
            bond_embedding = self.feat_drop(bond_embedding)

            graph.edata['bond_embedding'] = self.v(bond_embedding).view(-1, self.num_head, self.out_feat)

            atom_h = self.feat_drop(node_embedding)

            graph.ndata['k'] = self.k(atom_h).view(-1, self.num_head, self.out_feat)
            graph.ndata['q'] = self.q(atom_h).view(-1, self.num_head, self.out_feat)
            graph.apply_edges(fn.u_dot_v('k', 'q', 'e'))
            att = graph.edata.pop('e')
            dist_decay = graph.edata["basic_attn"].unsqueeze(dim=1)
            dist_decay = self.dis_lin(dist_decay).permute(0, 2, 1)
            sqrt_d = torch.sqrt(torch.tensor([self.out_feat]).to(devices))
            att = att.sum(dim=-1).unsqueeze(dim=-1) / sqrt_d  # (num_edge, num_heads, 1)
            att_decay = att + dist_decay
            att_decay = edge_softmax(graph, att_decay)  # 这里就是计算节点i所有相邻入边的注意力权重

            graph.edata['att_decay'] = self.attn_drop(att_decay)
            graph.update_all(e_mul_e('bond_embedding', 'att_decay', 'm'),
                             fn.sum('m', 'ft'))

            he = graph.ndata['ft'].view(-1, self.node_dim)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([he, node_embedding, he - node_embedding], dim=-1))
                beta = beta.sigmoid()
                he = beta * node_embedding + (1 - beta) * he
            else:
                he = he + node_embedding

            he = self.ffn(he, devices)
            return he


class Bind(nn.Module):

    def __init__(self, num_head, feat_drop, attn_drop, num_convs, node_dim, edge_dim, activation="ReLU",
                 beta=True):
        super(Bind, self).__init__()

        self.num_convs = num_convs

        self.a2e_layers = nn.ModuleList()
        self.e2e_layers = nn.ModuleList()
        self.e2a_layers = nn.ModuleList()

        for i in range(num_convs):  # 边的维度更新成和节点维度一致
            if i == 0:
                self.a2e_layers.append(Atom2BondLayer(node_dim, edge_dim, feat_drop=feat_drop, activation=activation))
            else:
                self.a2e_layers.append(Atom2BondLayer(node_dim, node_dim, feat_drop=feat_drop, activation=activation))
            self.e2e_layers.append(Bond2BondLayer(node_dim, num_head=num_head, feat_drop=feat_drop,
                                                  attn_drop=attn_drop, activation=activation, beta=beta))
            self.e2a_layers.append(Bond2AtomLayer(node_dim, num_head=num_head, feat_drop=feat_drop,
                                                  attn_drop=attn_drop, activation=activation))

    def forward(self, g, index_kj, index_ji, idx_i, idx_j, idx_k, devices):
        bond_embedding = g.edata["edge_feature_h"]  # 图中边的信息
        atom_embedding = g.ndata["node_feature_h"]  # 图中节点的信息

        bond_embedding = self.a2e_layers[0](g, atom_embedding, bond_embedding)
        for layer_num in range(self.num_convs):
            # bond_embedding = self.a2e_layers[layer_num](g, atom_embedding, bond_embedding)
            bond_embedding = self.e2e_layers[layer_num](g, bond_embedding, index_kj, index_ji, idx_i, idx_j, idx_k,
                                                        devices)
            g.edata["edge_feature_h"] = bond_embedding
            atom_embedding = self.e2a_layers[layer_num](g, bond_embedding, atom_embedding, devices)
            g.ndata["node_feature_h"] = atom_embedding
        return atom_embedding


class GPAW(nn.Module):

    def __init__(self, p_dropout=0.5, atom_dim=256, hidden_dim=128, edge_dim=16, node_dim=512,
                 in_dim_res=60, in_dim_edge=4, encoder_type="Bind", degree_information=0, GCN_=0,gpaw_layer_num = 1):
        super(GPAW, self).__init__()

        self.encoder_type = encoder_type
        self.degree_information = degree_information
        self.GCN_ = GCN_
        self.gpaw_layer_num =  gpaw_layer_num
        self.hidden_dim = hidden_dim
        self.bias = False

        self.gpaw_layers_a2a =  nn.ModuleList()
        self.gpaw_layers_other =  nn.ModuleList()

        for i in range(gpaw_layer_num):
            if i==0:
                self.gpaw_layers_a2a.append(N2N(in_channels=atom_dim+hidden_dim, hidden_channels=node_dim,num_hop=3, dropout=0.5, activation='relu',
                     feature_inter='attention', inter_layer=2, feature_fusion='attention', norm_type='ln'))
                self.gpaw_layers_other.append(
                    Bind(num_head=4, feat_drop=0.5, attn_drop=0, num_convs=2, node_dim=512, edge_dim=16))
            else:
                self.gpaw_layers_a2a.append(N2N(in_channels=node_dim, hidden_channels=node_dim, num_hop=3, dropout=0.5,activation='relu',
                           feature_inter='attention', inter_layer=2, feature_fusion='attention', norm_type='ln'))
                self.gpaw_layers_other.append(
                    Bind(num_head=4, feat_drop=0.5, attn_drop=0, num_convs=2, node_dim=512, edge_dim=512))
        self.pool = AtomPooling(atom_dim)
        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.lin_res1 = nn.Linear(in_dim_res, hidden_dim)
        self.lin_edge1 = nn.Linear(in_dim_edge, edge_dim)


    def triplets(self, g):  # g中有边的信息，点的信息
        row, col = g.edges()  # j --> i
        num_nodes = g.num_nodes()

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col.long(), col=row.long(), value=value, sparse_sizes=(num_nodes, num_nodes))# 邻接矩阵  row是入边，col是出边
        adj_t_row = adj_t[row.long()]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, g1, atom_feature_h, devices):
        metal_pos = ''
        mut_pos = ''
        for i in range(g1.ndata["res_feature_h"].shape[0]):
            if g1.ndata["res_feature_h"][i][-1] == 1:
                metal_pos = i
            if g1.ndata["res_feature_h"][i][-2] == 1:
                mut_pos = i

        i1, j1, idx_i1, idx_j1, idx_k1, idx_kj1, idx_ji1 = self.triplets(g1)

        g1.ndata["res_feature_h"] = self.lin_res1(g1.ndata["res_feature_h"])  # 氨基酸特征由60 到 128
        atom_feature_h = self.pool(atom_feature_h, g1.ndata["res_index_atom"])  # 256  原子特征
        g1.edata["edge_feature_h"] = self.lin_edge1(g1.edata["edge_feature_h"])  # 4--> 16 边的特征
        g1.ndata["node_feature_h"] = torch.cat((g1.ndata["res_feature_h"], atom_feature_h),1)  # 氨基酸特征拼接原子特征


        for layer_num in range(self.gpaw_layer_num):
            g1.ndata["node_feature_h"] = self.gpaw_layers_a2a[layer_num](g1, g1.ndata["node_feature_h"], devices)
            g1.ndata["node_feature_h"] = self.gpaw_layers_other[layer_num](g1, idx_kj1, idx_ji1, idx_i1, idx_j1, idx_k1, devices)

        h1 = g1.ndata["node_feature_h"]
        # metal = h1[metal_pos]
        # end_fea = torch.cat((metal, h1[mut_pos]), dim=0)
        end_h1 = h1[mut_pos]
        return end_h1

