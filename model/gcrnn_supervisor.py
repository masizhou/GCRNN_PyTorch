import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter  # pytorch的可视化工具Tensorboard

from lib import utils
from model.gcrnn_model import GCRNNModel
from model.loss import masked_mae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data') # train\test\val的batch_size 以及 数据集的路径
        self._model_kwargs = kwargs.get('model') # 关于模型的超参数，入输入输出等
        self._train_kwargs = kwargs.get('train') # 训练时的超参数，如学习率、训练次数等等
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.) # 梯度裁剪：self.max_grad_norm为 梯度的最大范数，应该是防止了梯度爆炸

        # logging.
        self._log_dir = self._get_log_dir(kwargs) # 日志文件夹
        # SummaryWriter的作用就是，将数据以特定的格式存储到上面得到的那个日志文件夹中
        self._writer = SummaryWriter('runs/' + self._log_dir) # 这个是先实例化对象，参数是文件夹的路径

        log_level = self._kwargs.get('log_level', 'INFO')  # 第二个参数为找不到键时返回的值
        # __name__在当前.py文件调用就是__main__；在其他的文件中调用，那就是当前文件的名字，也就是dcrnn_supervisor
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level) # 得到日志,level为日志的级别

        # data set
        self._data = utils.load_dataset(**self._data_kwargs) # 加载数据集
        self.standard_scaler = self._data['scaler'] # 标准化类的对象，可以调用transform方法标准化数据，inverse_transform恢复数据

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1)) # 数据集的节点个数
        self.input_dim = int(self._model_kwargs.get('input_dim', 1)) # 输入维度
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder 其实就是时间窗口的大小 为 12
        self.output_dim = int(self._model_kwargs.get('output_dim', 1)) # 模型的输出维度，为1维

        # todo 这个参数不明白
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))  # 这个参数是啥作用呢？？？
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder 解码的时候 预测的范围

        # setup model
        gcrnn_model = GCRNNModel(adj_mx, self._logger, **self._model_kwargs) # 实例化一个GCRNN模型对象
        self.gcrnn_model = gcrnn_model.cuda() if torch.cuda.is_available() else gcrnn_model # 将模型放到GPU上（如果可用的话）
        self._logger.info("Model created") # 在终端打印 模型被创建

        self._epoch_num = self._train_kwargs.get('epoch', 0) # 这个参数就是，当为0时，训练模型，大于0时：就加载对应保存的那个epoch模型
        if self._epoch_num > 0: # 如果设置epoch这个参数>0,则直接加载模型
            self.load_model() # 写了个加载模型的方法

    @staticmethod
    def _get_log_dir(kwargs): # 前导单下划线表示：不要在类外访问（类似就私有方法，但只是给程序员的警告，实际能访问）
        """
        作用：创建日志文件夹
        """
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step') # 最大的扩散步数
            num_rnn_layers = kwargs['model'].get('num_rnn_layers') # rnn的层数
            rnn_units = kwargs['model'].get('rnn_units') # 隐层神经元的个数
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type') # 滤波器的类型
            filter_type_abbr = 'L' # 滤波器类型的缩写
            if filter_type == 'random_walk': # 随机游走
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk': # 双向的随机游走
                filter_type_abbr = 'DR'
            run_id = 'gcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id) # 拼接成日志的路径
        if not os.path.exists(log_dir): # 若不存在，则根据前面拼接的路径创建
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        """
        作用：保存模型
        """
        if not os.path.exists('models/'): # 不存在此路径，则创建
            os.makedirs('models/')

        config = dict(self._kwargs) # 将所有的参数转换成字典（其实已经是字典了，这里应该是再保证一下）
        config['model_state_dict'] = self.gcrnn_model.state_dict() # 存储网络结构的名字和对应的参数
        config['epoch'] = epoch # 训练时 第几轮
        torch.save(config, 'models/epo%d.tar' % epoch) # 保存模型，第一个参数为 要保存的对象；第二个参数为 保存文件名
        self._logger.info("Saved model at {}".format(epoch)) # 保存日志
        return 'models/epo%d.tar' % epoch # 将保存的模型的文件路径返回

    def load_model(self):
        """
        作用：加载模型
        """
        self._setup_graph() # ？？？？？？
        # 断言：要是路径存在 则继续，要是找不到 则抛出异常，显示后面的错误提示
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu') # 加载模型参数
        self.gcrnn_model.load_state_dict(checkpoint['model_state_dict']) # 将模型参数 加载到模型中
        self._logger.info("Loaded model at {}".format(self._epoch_num)) # 输出日志

    def _setup_graph(self):
        """
        作用：？？？？？？
        """
        with torch.no_grad(): # 不记录梯度
            self.gcrnn_model = self.gcrnn_model.eval() # 打开测试模式

            val_iterator = self._data['val_loader'].get_iterator() # 加载验证数据集

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y) # 将数据 转换成 符合dcrnn模型的数据
                output = self.gcrnn_model(x) # 将数据送入模型，得出预测值
                break

    def train(self, **kwargs):
        """
        作用：模型训练的入口
        """
        kwargs.update(self._train_kwargs) # 把self._train_kwargs字典 的键值对 更新到 kwargs 中
        return self._train(**kwargs) # 返回训练的结果

    def evaluate(self, dataset='val', batches_seen=0):
        """
        作用：计算预测结果的损失
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.gcrnn_model = self.gcrnn_model.eval() # 打开测试模式

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator() # 加载验证数据集，这是个迭代器
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.gcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses) # 平均损失

            # 将之前定义的self._writer对象，调用.add_scalar方法，其作用是：存放数据到指定的文件夹中
            # 作用：tensorboard可视化使用
            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen) # 变量名、存放的值、x坐标

            # 因为前面一个batch一个batch的把数据加入到列表的，因此下面就在batch的维度上拼接起来
            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            # 前面的预测值和真实值，都是标准化之后的，下面是恢复之后的数据
            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth) # 恢复为真实的 真实值
                y_preds_scaled.append(y_pred) # 恢复为真实的 预测值

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        """
        作用：训练模型
        patience：这个参数的作用就是，当训练50次，每次的验证训练损失都比前一次大，那就终止训练
        """
        min_val_loss = float('inf') # 最小的验证损失 设置为 正无穷
        wait = 0
        optimizer = torch.optim.Adam(self.gcrnn_model.parameters(), lr=base_lr, eps=epsilon) # Adam优化器

        # 学习率的调整，参考：https://zhuanlan.zhihu.com/p/69411064
        #  下面这个：按设定的间隔调整学习率。这个方法适合后期调试使用，观察loss曲线，为每个实验定制学习率调整时机。
        # 参数：优化器、调整学习率的间隔（list，递增）、学习率调整倍数（这里默认了0.1,也就是10倍）
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...') # 打印开始 训练 的日志

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch # 一共需要的batch的个数
        self._logger.info("num_batches:{}".format(num_batches)) # 记录日志

        batches_seen = num_batches * self._epoch_num # 训练一轮需要的batch个数，再乘训练的次数？??

        for epoch_num in range(self._epoch_num, epochs):

            self.gcrnn_model = self.gcrnn_model.train() # 打开训练模式

            train_iterator = self._data['train_loader'].get_iterator() # 加载训练数据集，返回的是一个生成器
            losses = [] # 要保存的loss

            start_time = time.time() # 一轮训练的开始时间

            for _, (x, y) in enumerate(train_iterator): # train_iterator是一个生成器，可以迭代，每次迭代则切片 取一次数据，好处是节约内存啊
                optimizer.zero_grad() # 梯度清零

                x, y = self._prepare_data(x, y) # 将数据转换为 可以送入模型的数据

                output = self.gcrnn_model(x, y, batches_seen) # 调用模型，得出预测结果，其中batches_seen为：迄今为止 经过的 批次

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.gcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                loss = self._compute_loss(y, output) # 计算带mask的mae损失

                self._logger.debug(loss.item()) # 记录日志

                losses.append(loss.item()) # 将损失的数值，加入到列表中

                batches_seen += 1 # 经过的 批次 +1
                loss.backward() # 反向传播，计算梯度

                # gradient clipping - this does it in place 梯度裁剪
                """
                梯度裁剪：self.max_grad_norm为 梯度的最大范数，应该是防止了梯度爆炸
                """
                torch.nn.utils.clip_grad_norm_(self.gcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step() # 更新参数
            self._logger.info("epoch complete")
            lr_scheduler.step() # 更新学习率
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen) # 计算验证集的损失，返回loss、真实值、预测值

            end_time = time.time() # 训练一个epoch的结束时间

            # 下面是：将训练的loss加入到tensorboard的可视化文件中
            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1: # 每log_every记录一次日志
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message) # 日志

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1: # 每10个epoch进行一次测试（测试集）
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message) # 日志

            if val_loss < min_val_loss: # 若当前的验证损失 < 最小验证损失，则保存当前的模型参数
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num) # 保存模型，并得到保存的模型的文件路径及文件名
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss # 更新验证损失 为 当前验证损失

            elif val_loss >= min_val_loss: # 若当前的验证损失 >= 最小验证损失
                wait += 1
                if wait == patience: # 这个就是说，当训练了patience次，每次的验证训练损失都比前一次大，那就终止训练
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        """
        作用：准备数据（数据维度的转变，以及放到GPU上等操作）
        """
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device) # 将数据放到GPU上，并返回

    def _get_x_y(self, x, y):
        """
        作用：将数据转换为tensor,并转换成对应的维度
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float() # 转换成tensor
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size())) # 记录日志：x的维度
        self._logger.debug("y: {}".format(y.size())) # 记录日志：y的维度
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        作用：调整x,y的维度
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        # 将y中最里面的维度进行切片取前output，之后再将最后两维合并成一维（这两维为：节点个数和输出特征）
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        """
        作用：计算损失
        """
        y_true = self.standard_scaler.inverse_transform(y_true) # 之前不是标准化了么，现在恢复数据
        y_predicted = self.standard_scaler.inverse_transform(y_predicted) # 把预测值也恢复了
        return masked_mae_loss(y_predicted, y_true) # 调用了计算带mask的mae损失的函数
