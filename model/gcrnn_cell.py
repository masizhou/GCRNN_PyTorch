import numpy as np
import torch

from lib import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        """
        作用：定义属性，初始化参数和偏置
        """
        self._rnn_network = rnn_network # rnn网络
        self._params_dict = {} # 存放参数的字典
        self._biases_dict = {} # 存放偏置的字典
        self._type = layer_type # 每层的类型

    def get_weights(self, shape):
        """
        作用：根据给出的shape，初始化权重参数
        """
        if shape not in self._params_dict:
            # torch.nn.Parameter():是一个Tensor，也就是说Tensor 拥有的属性它都有
            # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter；
            # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param) # 前面把nn_param用0填充，现在再用正态分布的方式初始化
            self._params_dict[shape] = nn_param # 加入到参数字典中

            # .register_parameter()作用和.Parameter()一样，只不过是向 我们建立的网络module添加 parameter
            # 第一个参数为参数名字，第二个参数为Parameter()对象，其实是个Tensor矩阵
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        """
        作用：根据长度 初始化偏置
        """
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start) # 用值bias_start填充向量biases。
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class GCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=False):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self.adj_mx = adj_mx
        self._num_units = num_units # 隐层的个数
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru # 是否使用图卷积计算重置门r 和 更新门u
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx, lambda_max=None))
        elif filter_type == "attention":
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc') # 这里的self就是DCGRUCell
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    @staticmethod
    def cheb_polynomial(laplacian, K):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([K, N, N], device=device)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=device)  # 0阶的切比雪夫多项式为单位阵

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2: # 1阶切比雪夫多项式就是拉普拉斯矩阵 L 本身
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2] #切比雪夫多项式的递推式:T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(adj):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        D = torch.diag(torch.sum(adj, dim=-1) ** (-1 / 2))
        D = torch.where(torch.isinf(D), torch.full_like(D, 0), D)
        L = torch.eye(adj.size(0), device=adj.device) - torch.mm(torch.mm(D, adj), D) # L = I - D^-1/2 * A * D^-1/2
        return L


    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.带有图卷积的门控循环单元(GRU)
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv # can use other graph conv?
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units) # can use other graph conv?
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.matmul(inputs_and_state, weights)
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    # dcrnn
    def _dconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    # gcn
    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)

        for support in self._supports:
            x1 = torch.sparse.mm(support, x0) # L * X
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    # chebnet
    def _chebconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)

        L = self.get_laplacian(torch.tensor(self.adj_mx, device=device)) # [N, N]
        mul_L = self.cheb_polynomial(L, K=2) # [K, N, N]
        # print(mul_L.shape, x0.shape) # torch.Size([2, 82, 82]) torch.Size([82, 8320])
        for _ in range(len(self._supports)):
            x1 = torch.matmul(mul_L, x0) # (k, num_nodes, input_size * batch_size)
            x1 = torch.sum(x1, dim=0) # (num_nodes, input_size * batch_size)
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
