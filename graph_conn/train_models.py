import torch
from torch import nn
from graph_conn.models import GCNNet
from dgl.dataloading import GraphDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from graph_conn import models
from graph_conn.models import NetParams
import dgl


class ConnGCM:
    """
       Model class for fitting data

       Methods:

       make_loader(): returns DataLoader

       train(): performs model training

    """

    def __init__(self, net_params):
        self.net_params = net_params
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = GCNNet(net_params=net_params).to(self.device)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def _init_optimizer(self):
        return torch.optim.Adam(self.net.parameters(),
                                lr=self.net_params.init_lr, weight_decay=self.net_params.weight_decay)

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                          factor=self.net_params.lr_reduce_factor,
                                                          patience=self.net_params.lr_schedule_patience,
                                                          verbose=True)

    def _init_tensorboard(self, logdir=None, comment=''):
        return SummaryWriter(log_dir=logdir, comment=comment)

    def train_epoch(self, dataloader):
        self.net.train()
        epoch_loss = 0
        epoch_train_acc = 0
        for iter, (batch_graphs, batch_labels) in enumerate(dataloader):
            batch_feat = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_eweight = batch_graphs.edata['weight'].to(self.device)
            batch_labels = batch_labels.to(self.device)
            self.optimizer.zero_grad()
            scores = self.net(batch_graphs, batch_feat, batch_eweight)
            loss = self.criterion(scores, batch_labels)
            epoch_loss += loss.detach().item()
            scores = scores.detach().argmax(dim=1)
            epoch_train_acc += (scores == batch_labels).float().sum().item()
        epoch_loss /= (iter + 1)
        epoch_train_acc /= (iter + 1)
        return epoch_loss, epoch_train_acc

    def eval_epoch(self, dataloader):
        self.net.eval()
        epoch_test_loss = 0
        epoch_test_acc = 0
        with torch.no_grad():
            for iter, (batch_graphs, batch_labels) in enumerate(dataloader):
                batch_feat = batch_graphs.ndata['feat'].to(self.device)  # num x feat
                batch_eweight = batch_graphs.edata['weight'].to(self.device)
                batch_labels = batch_labels.to(self.device)
                scores = self.net(batch_graphs,batch_feat,batch_eweight)
                loss = self.criterion(scores, batch_labels)
                epoch_test_loss += loss.detach().item()
                scores = scores.detach().argmax(dim=1)
                epoch_test_acc += (scores == batch_labels).float().sum().item()
            epoch_test_loss /= (iter + 1)
            epoch_test_acc /= (iter + 1)
            return epoch_test_loss, epoch_test_acc

    def train(self, dataset, scheduler=False):

        train_loader, test_loader, val_loader = self.make_loader(dataset)
        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
                val_loss {v_loss:0.4f}"
        with tqdm(desc="epoch", total=self.net_params.n_epochs) as pbar_outer:
            for epoch in range(self.net_params.n_epochs):
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.eval_epoch(val_loader)
                history.append([(train_loss, train_acc), (val_loss, val_acc)])

                if scheduler:
                    self.scheduler.step(val_loss)

                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                               v_loss=val_loss))
        return history

    def make_loader(self, dataset):
        """

        Args:
            dataset: dataset instance from conn_dataset

        Returns:

        """
        split = dataset.get_split_idx(test_size=self.net_params.test_size, val_size=self.net_params.val_size)
        test_dataset = dataset[split['test']]
        train_dataset = dataset[split['train']]
        val_dataset = dataset[split['train']]
        train_loader = GraphDataLoader(train_dataset, batch_size=self.net_params.batch_size)
        test_loader = GraphDataLoader(test_dataset, batch_size=self.net_params.batch_size)
        val_loader = GraphDataLoader(val_dataset, batch_size=self.net_params.batch_size)
        return train_loader, test_loader, val_loader
