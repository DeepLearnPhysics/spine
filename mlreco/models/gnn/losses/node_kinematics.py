import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.bayes.evidential import EDLRegressionLoss, EVDLoss

class LogRMSE(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none', eps=1e-7):
        super(LogRMSE, self).__init__()
        self.reduction = reduction
        self.mseloss = torch.nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, inputs, targets):
        x = torch.log(inputs + self.eps)
        y = torch.log(targets + self.eps)
        out = self.mseloss(x, y)
        out = torch.sqrt(out + self.eps)
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            return out


class BerHuLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none'):
        super(BerHuLoss, self).__init__()
        self.reduction == reduction

    def forward(self, inputs, targets):
        norm = torch.abs(inputs - targets)
        c = norm.max() * 0.20
        out = torch.where(norm <= c, norm, (norm**2 + c**2) / (2.0 * c))
        if self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'mean':
            return out.mean()
        else:
            return out


class NodeKinematicsLoss(torch.nn.Module):
    """
    Takes the n-features node output of the GNN and optimizes
    node-wise scores such that the score corresponding to the
    correct class is maximized.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        grappa_loss:
          node_loss:
            name:           : type
            batch_col       : <column in the label data that specifies the batch ids of each voxel (default 3)>
            type_col        : <column in the label data that specifies the target node class (default 7)>
            momentum_col    : <column in the label data that specifies the target node momentum (default 8)>
            loss            : <loss function: 'CE' or 'MM' (default 'CE')>
            reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
            balance_classes : <balance loss per class: True or False (default False)>
    """
    def __init__(self, loss_config):
        super(NodeKinematicsLoss, self).__init__()

        # Set the target for the loss
        self.batch_col = loss_config.get('batch_col', 3)
        self.type_col = loss_config.get('target_col', 7)
        self.momentum_col = loss_config.get('target_col', 8)
        self.vtx_col = loss_config.get('vtx_col', 9)
        self.vtx_positives_col = loss_config.get('vtx_positives_col', 12)

        # Set the losses
        self.type_loss = loss_config.get('type_loss', 'CE')
        self.reg_loss = loss_config.get('reg_loss', 'l2')
        self.reduction = loss_config.get('reduction', 'sum')
        self.balance_classes = loss_config.get('balance_classes', False)
        self.spatial_size = loss_config.get('spatial_size', 768)

        if self.type_loss == 'CE':
            self.type_lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=-1)
        elif self.type_loss == 'EVD':
            evd_loss_name = loss_config.get('evd_loss_name', 'evd_nll')
            T = loss_config.get('T', 50000)
            self.type_lossfn = EVDLoss(evd_loss_name, reduction=self.reduction,T=T, one_hot=False, mode='evidence')
        elif self.type_loss == 'MM':
            p = loss_config.get('p', 1)
            margin = loss_config.get('margin', 1.0)
            self.type_lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise ValueError('Type loss not recognized: ' + self.type_loss)

        if self.reg_loss == 'l2':
            self.reg_lossfn = torch.nn.MSELoss(reduction=self.reduction)
        elif self.reg_loss == 'edl':
            w = loss_config.get('kld_weight', 0.0)
            self.reg_lossfn = EDLRegressionLoss(reduction=self.reduction, w=w)
        elif self.reg_loss == 'l1':
            self.reg_lossfn = torch.nn.L1Loss(reduction=self.reduction)
        elif self.reg_loss == 'log_rmse':
            self.reg_lossfn = LogRMSE(reduction=self.reduction)
        elif self.reg_loss == 'huber':
            self.reg_lossfn = torch.nn.SmoothL1Loss(reduction=self.reduction)
        elif self.reg_loss == 'berhu':
            self.reg_lossfn = BerHuLoss(reduction=self.reduction)
        else:
            raise ValueError('Regression loss not recognized: ' + self.reg_loss)

        self.vtx_position_loss = torch.nn.L1Loss(reduction='none')
        self.vtx_score_loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, out, types):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            types ([torch.tensor])     : (N,9) [x, y, z, batchid, value, id, groupid, pdg, p]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        type_loss, p_loss, type_acc, p_acc = 0., 0., 0., 0.
        vtx_position_loss, vtx_score_loss, vtx_position_acc, vtx_score_acc = 0., 0., 0., 0.
        n_clusts_type, n_clusts_momentum, n_clusts_vtx, n_clusts_vtx_positives = 0, 0, 0, 0

        compute_type = 'node_pred_type' in out
        compute_momentum = 'node_pred_p' in out
        compute_vtx = 'node_pred_vtx' in out

        for i in range(len(types)):

            # If the input did not have any node, proceed
            if not compute_type and not compute_momentum and not compute_vtx:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:,self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                clusts = out['clusts'][i][j]

                # Increment the type loss, balance classes if requested
                if compute_type:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_type = out['node_pred_type'][i][j]
                    if not node_pred_type.shape[0]:
                        continue
                    node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)
                    node_assn_type = torch.tensor(node_assn_type, dtype=torch.long, device=node_pred_type.device, requires_grad=False)

                    # Do not apply loss to nodes labeled -1 (unknown class)
                    node_mask = torch.nonzero(node_assn_type > -1, as_tuple=True)[0]
                    if len(node_mask):
                        node_pred_type = node_pred_type[node_mask]
                        node_assn_type = node_assn_type[node_mask]

                        if self.balance_classes:
                            vals, counts = torch.unique(node_assn_type, return_counts=True)
                            weights = np.array([float(counts[k])/len(node_assn_type) for k in range(len(vals))])
                            for k, v in enumerate(vals):
                                loss = (1./weights[k])*self.type_lossfn(node_pred_type[node_assn_type==v], node_assn_type[node_assn_type==v])
                                total_loss += loss
                                type_loss += float(loss)
                        else:
                            loss = self.type_lossfn(node_pred_type, node_assn_type)
                            total_loss += loss
                            type_loss += float(loss)

                        # Increment the number of nodes
                        n_clusts_type +=len(node_mask)

                # Increment the momentum loss
                if compute_momentum:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_p = out['node_pred_p'][i][j]
                    if not node_pred_p.shape[0]:
                        continue
                    node_assn_p = get_momenta_label(labels, clusts, column=self.momentum_col)
                    loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
                    total_loss += loss
                    p_loss += float(loss)

                    # Increment the number of nodes
                    n_clusts_momentum += len(clusts)

                if compute_vtx:
                    node_pred_vtx = out['node_pred_vtx'][i][j]
                    if not node_pred_vtx.shape[0]:
                        continue

                    # Predictions are shifts w.r.t the barycenter of each cluster
                    # anchors = []
                    # for c in clusts:
                    #     anchors.append(torch.mean(labels[c, :3], dim=0) + 0.5)
                    # anchors = torch.stack(anchors)
                    # node_pred_vtx[:, :3] = node_pred_vtx[:, :3] + anchors

                    node_x_vtx = get_cluster_label(labels, clusts, column=self.vtx_col)
                    node_y_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+1)
                    node_z_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+2)

                    node_assn_vtx = torch.tensor(np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1),
                                                dtype=torch.float, device=node_pred_vtx.device, requires_grad=False)
                    node_assn_vtx = node_assn_vtx/self.spatial_size

                    # Exclude vertex that is outside of the volume
                    good_index = torch.all(torch.abs(node_assn_vtx) <= 1., dim=1)

                    #positives = get_cluster_label(labels, clusts, column=self.vtx_positives_col)
                    # Take the max for each cluster - e.g. for a shower, the primary fragment only
                    # is marked as primary particle, so taking majority count would eliminate the shower
                    # from primary particles for vertex identification purpose.
                    positives = []
                    for c in clusts:
                        positives.append(labels[c, self.vtx_positives_col].max().item())
                    positives = np.array(positives)

                    positives = torch.tensor(positives, dtype=torch.long, device=node_pred_vtx.device, requires_grad=False)
                    # for now only sum losses, they get averaged below in results dictionary
                    loss2 = self.vtx_score_loss(node_pred_vtx[good_index, 3:], positives[good_index])
                    loss1 = torch.sum(torch.mean(self.vtx_position_loss(node_pred_vtx[good_index & positives.bool(), :3], node_assn_vtx[good_index & positives.bool()]), dim=1))

                    total_loss += loss1 + loss2

                    vtx_position_loss += float(loss1)
                    vtx_score_loss += float(loss2)

                    n_clusts_vtx += (good_index).sum().item()
                    n_clusts_vtx_positives += (good_index & positives.bool()).sum().item()
                    # print("Removing", (~good_index).sum().item(), len(good_index) )

                # Compute the accuracy of assignment (fraction of correctly assigned nodes)
                # and the accuracy of momentum estimation (RMS relative residual)
                if compute_type:
                    type_acc += float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))

                if compute_momentum:
                    p_acc += float(torch.sum(1.- torch.abs(node_pred_p.squeeze()-node_assn_p)/node_assn_p)) # 1-MAPE

                if compute_vtx and node_pred_vtx[good_index].shape[0]:
                    # print(node_pred_vtx[good_index & positives.bool(), :3], node_assn_vtx[good_index & positives.bool()])
                    vtx_position_acc += float(torch.sum(1. - torch.abs(node_pred_vtx[good_index & positives.bool(), :3]-node_assn_vtx[good_index & positives.bool()])/(torch.abs(node_assn_vtx[good_index & positives.bool()]) + torch.abs(node_pred_vtx[good_index & positives.bool(), :3]))))/3.
                    vtx_score_acc += float(torch.sum(torch.argmax(node_pred_vtx[good_index, 3:], dim=1) == positives[good_index]))

        n_clusts = n_clusts_type + n_clusts_momentum + n_clusts_vtx + n_clusts_vtx_positives

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            result = {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=types[0].device, dtype=torch.float),
                'n_clusts_momentum': n_clusts_momentum,
                'n_clusts_type': n_clusts_type,
                'n_clusts_vtx': n_clusts_vtx,
                'n_clusts_vtx_positives': n_clusts_vtx_positives
            }
            if compute_type:
                result.update({
                    'type_loss': 0.,
                    'type_accuracy': 0.,
                })
            if compute_momentum:
                result.update({
                    'p_loss': 0.,
                    'p_accuracy': 0.,
                })
            if compute_vtx:
                result.update({
                    'vtx_position_loss': 0.,
                    'vtx_score_loss': 0.,
                    'vtx_position_acc': 0.,
                    'vtx_score_acc': 0.,
                })
            return result

        result = {
            'accuracy': (type_acc+p_acc+vtx_position_acc+vtx_score_acc)/n_clusts,
            'loss': total_loss/n_clusts,
            'n_clusts_momentum': n_clusts_momentum,
            'n_clusts_type': n_clusts_type,
            'n_clusts_vtx': n_clusts_vtx,
            'n_clusts_vtx_positives': n_clusts_vtx_positives
        }

        if compute_type:
            result.update({
                'type_accuracy': 0. if not n_clusts_type else type_acc/n_clusts_type,
                'type_loss': 0. if not n_clusts_type else type_loss/n_clusts_type,
            })
        if compute_momentum:
            result.update({
                'p_accuracy': 0. if not n_clusts_momentum else p_acc/n_clusts_momentum,
                'p_loss': 0. if not n_clusts_momentum else p_loss/n_clusts_momentum,
            })
        if compute_vtx:
            result.update({
                'vtx_score_loss': 0. if not n_clusts_vtx else vtx_score_loss/n_clusts_vtx,
                'vtx_score_acc': 0. if not n_clusts_vtx else vtx_score_acc/n_clusts_vtx,
                'vtx_position_loss': 0. if not n_clusts_vtx_positives else vtx_position_loss/n_clusts_vtx_positives,
                'vtx_position_acc': 0. if not n_clusts_vtx_positives else vtx_position_acc/n_clusts_vtx_positives,
            })

        return result


class NodeEvidentialKinematicsLoss(NodeKinematicsLoss):

    def __init__(self, loss_config):
        super(NodeEvidentialKinematicsLoss, self).__init__(loss_config)
        evd_loss_name = loss_config.get('evd_loss_name', 'evd_nll')
        T = loss_config.get('T', 50000)
        self.type_lossfn = EVDLoss(evd_loss_name, 
                                   reduction='sum',
                                   T=T, 
                                   one_hot=False, 
                                   mode='evidence')
        w = loss_config.get('kld_weight', 0.0)
        if self.reg_loss == 'l2':
            self.reg_lossfn = torch.nn.MSELoss(reduction=self.reduction)
        elif self.reg_loss == 'edl':
            w = loss_config.get('kld_weight', 0.0)
            kl_mode = loss_config.get('kl_mode', 'evd')
            self.reg_lossfn = EDLRegressionLoss(reduction=self.reduction, w=w, kl_mode=kl_mode)
        elif self.reg_loss == 'l1':
            self.reg_lossfn = torch.nn.L1Loss(reduction=self.reduction)
        elif self.reg_loss == 'log_rmse':
            self.reg_lossfn = LogRMSE(reduction=self.reduction)
        elif self.reg_loss == 'huber':
            self.reg_lossfn = torch.nn.SmoothL1Loss(reduction=self.reduction)
        elif self.reg_loss == 'berhu':
            self.reg_lossfn = BerHuLoss(reduction=self.reduction)
        else:
            raise ValueError('Regression loss not recognized: ' + self.reg_loss)

        self.compute_momentum_switch = loss_config.get('compute_momentum', True)

    def compute_type(self, node_pred_type, labels, clusts, iteration):
        '''
        
        '''
        # assert False
        total_loss, n_clusts_type, type_loss = 0, 0, 0

        node_assn_type = get_cluster_label(labels, clusts, column=self.type_col)
        node_assn_type = torch.tensor(node_assn_type, dtype=torch.long, 
                                                      device=node_pred_type.device, 
                                                      requires_grad=False)

        # Do not apply loss to nodes labeled -1 (unknown class)
        node_mask = torch.nonzero(node_assn_type > -1, as_tuple=True)[0]
        if len(node_mask):
            node_pred_type = node_pred_type[node_mask]
            node_assn_type = node_assn_type[node_mask]

            if self.balance_classes:
                vals, counts = torch.unique(node_assn_type, return_counts=True)
                weights = np.array([float(counts[k]) / len(node_assn_type) for k in range(len(vals))])
                for k, v in enumerate(vals):
                    loss = (1. / weights[k]) * self.type_lossfn(node_pred_type[node_assn_type==v], 
                                                                node_assn_type[node_assn_type==v],
                                                                t=iteration)
                    total_loss += loss
                    type_loss += float(loss)
            else:
                loss = self.type_lossfn(node_pred_type, node_assn_type)
                total_loss += torch.clamp(loss, min=0)
                type_loss += float(torch.clamp(loss, min=0))

            # Increment the number of nodes
            n_clusts_type += len(node_mask)
        
        with torch.no_grad():
            type_acc = float(torch.sum(torch.argmax(node_pred_type, dim=1) == node_assn_type))

        # print("TYPE: {}, {}, {}, {}".format(
        #     float(total_loss), float(type_loss), float(n_clusts_type), float(type_acc)))

        return total_loss, type_loss, n_clusts_type, type_acc


    def compute_momentum(self, node_pred_p, labels, clusts, iteration=None):

        node_assn_p = get_momenta_label(labels, clusts, column=self.momentum_col)
        with torch.no_grad():
            p_acc = torch.pow(node_pred_p[:, 0]-node_assn_p, 2).sum()
        n_clusts_momentum = len(clusts)
        
        if self.reg_loss == 'edl':
            loss, nll_loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
            nll_loss = torch.clamp(nll_loss, min=0).detach().cpu().numpy()
            p_acc = np.exp(-nll_loss)
            loss, nll_loss, p_acc = loss.nansum(), float(np.nansum(nll_loss)), np.sum(p_acc)
            # print("Momentum: {}, {}, {}, {}".format(
            #     float(loss), float(nll_loss), float(n_clusts_momentum), float(p_acc)))
            return loss, float(nll_loss), n_clusts_momentum, p_acc
        else:
            loss = self.reg_lossfn(node_pred_p.squeeze(), node_assn_p.float())
            return loss, float(loss), n_clusts_momentum, p_acc


    def compute_vertex(self, node_pred_vtx, labels, clusts):
        node_x_vtx = get_cluster_label(labels, clusts, column=self.vtx_col)
        node_y_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+1)
        node_z_vtx = get_cluster_label(labels, clusts, column=self.vtx_col+2)

        node_assn_vtx = torch.tensor(np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1),
                                    dtype=torch.float, device=node_pred_vtx.device, requires_grad=False)
        node_assn_vtx = node_assn_vtx/self.spatial_size

        # Exclude vertex that is outside of the volume
        good_index = torch.all(torch.abs(node_assn_vtx) <= 1., dim=1)

        #positives = get_cluster_label(labels, clusts, column=self.vtx_positives_col)
        # Take the max for each cluster - e.g. for a shower, the primary fragment only
        # is marked as primary particle, so taking majority count would eliminate the shower
        # from primary particles for vertex identification purpose.
        positives = []
        for c in clusts:
            positives.append(labels[c, self.vtx_positives_col].max().item())
        positives = np.array(positives)

        positives = torch.tensor(positives, dtype=torch.long, 
                                            device=node_pred_vtx.device, 
                                            requires_grad=False)
        # for now only sum losses, they get averaged below in results dictionary
        loss2 = self.vtx_score_loss(node_pred_vtx[good_index, 3:], positives[good_index])
        loss1 = torch.sum(torch.mean(
            self.vtx_position_loss(
                node_pred_vtx[good_index & positives.bool(), :3], 
                node_assn_vtx[good_index & positives.bool()]), dim=1))

        loss = loss1 + loss2

        vtx_position_loss = float(loss1)
        vtx_score_loss = float(loss2)

        n_clusts_vtx = (good_index).sum().item()
        n_clusts_vtx_positives = (good_index & positives.bool()).sum().item()

        if node_pred_vtx[good_index].shape[0]:
            # print(node_pred_vtx[good_index & positives.bool(), :3], node_assn_vtx[good_index & positives.bool()])
            vtx_position_acc = float(torch.sum(1. - torch.abs(node_pred_vtx[good_index & positives.bool(), :3] - \
                                                              node_assn_vtx[good_index & positives.bool()]) / \
                                    (torch.abs(node_assn_vtx[good_index & positives.bool()]) + \
                                     torch.abs(node_pred_vtx[good_index & positives.bool(), :3]))))/3.
            vtx_score_acc = float(torch.sum(
                torch.argmax(node_pred_vtx[good_index, 3:], dim=1) == positives[good_index]))
        
        out =  (loss, 
                vtx_position_loss, 
                vtx_score_loss, 
                n_clusts_vtx, 
                n_clusts_vtx_positives,
                vtx_position_acc,
                vtx_score_acc)
        
        return out


    def forward(self, out, types, iteration):
        """
        Applies the requested loss on the node prediction.

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            types ([torch.tensor])     : (N,9) [x, y, z, batchid, value, id, groupid, pdg, p]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        type_loss, p_loss, type_acc, p_acc = 0., 0., 0., 0.
        vtx_position_loss, vtx_score_loss, vtx_position_acc, vtx_score_acc = 0., 0., 0., 0.
        n_clusts_type, n_clusts_momentum, n_clusts_vtx, n_clusts_vtx_positives = 0, 0, 0, 0

        compute_type = 'node_pred_type' in out
        compute_momentum = 'node_pred_p' in out
        if not self.compute_momentum_switch:
            compute_momentum = False
        compute_vtx = 'node_pred_vtx' in out

        for i in range(len(types)):

            # If the input did not have any node, proceed
            if not compute_type and not compute_momentum and not compute_vtx:
                continue

            # Get the list of batch ids, loop over individual batches
            batches = types[i][:,self.batch_col]
            nbatches = len(batches.unique())
            for j in range(nbatches):

                # Narrow down the tensor to the rows in the batch
                labels = types[i][batches==j]

                clusts = out['clusts'][i][j]

                # Increment the type loss, balance classes if requested
                if compute_type:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_type = out['node_pred_type'][i][j]
                    if not node_pred_type.shape[0]:
                        continue
                    type_data = self.compute_type(node_pred_type, labels, clusts, iteration)
                    
                    total_loss    += type_data[0]
                    type_loss     += type_data[1]
                    n_clusts_type += type_data[2]
                    type_acc      += type_data[3]

                # Increment the momentum loss
                if compute_momentum:
                    # Get the class labels and true momenta from the specified columns
                    node_pred_p = out['node_pred_p'][i][j]
                    if not node_pred_p.shape[0]:
                        continue
                    p_data = self.compute_momentum(node_pred_p, labels, clusts, iteration)


                    total_loss        += p_data[0]
                    p_loss            += float(p_data[1])
                    n_clusts_momentum += p_data[2]
                    p_acc             += p_data[3] # Convert NLL loss to Model Evidence
                if compute_vtx:
                    node_pred_vtx = out['node_pred_vtx'][i][j]
                    if not node_pred_vtx.shape[0]:
                        continue
                    
                    vtx_data = self.compute_vertex(node_pred_vtx, labels, clusts)

                    total_loss             += vtx_data[0]
                    vtx_position_loss      += vtx_data[1]
                    vtx_score_loss         += vtx_data[2]
                    n_clusts_vtx           += vtx_data[3]
                    n_clusts_vtx_positives += vtx_data[4]
                    vtx_position_acc       += vtx_data[5]
                    vtx_score_loss         += vtx_data[6]

                # Compute the accuracy of assignment (fraction of correctly assigned nodes)
                # and the accuracy of momentum estimation (RMS relative residual)

        n_clusts = n_clusts_type + n_clusts_momentum + n_clusts_vtx + n_clusts_vtx_positives

        # Handle the case where no cluster/edge were found
        if not n_clusts:
            result = {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, 
                                         device=types[0].device, 
                                         dtype=torch.float),
                'n_clusts_momentum': n_clusts_momentum,
                'n_clusts_type': n_clusts_type,
                'n_clusts_vtx': n_clusts_vtx,
                'n_clusts_vtx_positives': n_clusts_vtx_positives
            }
            if compute_type:
                result.update({
                    'type_loss': 0.,
                    'type_accuracy': 0.,
                })
            if compute_momentum:
                result.update({
                    'p_loss': 0.,
                    'p_accuracy': 0.,
                })
            if compute_vtx:
                result.update({
                    'vtx_position_loss': 0.,
                    'vtx_score_loss': 0.,
                    'vtx_position_acc': 0.,
                    'vtx_score_acc': 0.,
                })
            return result

        result = {
            'accuracy': (type_acc+p_acc+vtx_position_acc+vtx_score_acc)/n_clusts,
            'loss': total_loss/n_clusts,
            'n_clusts_momentum': n_clusts_momentum,
            'n_clusts_type': n_clusts_type,
            'n_clusts_vtx': n_clusts_vtx,
            'n_clusts_vtx_positives': n_clusts_vtx_positives
        }

        if compute_type:
            result.update({
                'type_accuracy': 0. if not n_clusts_type else type_acc/n_clusts_type,
                'type_loss': 0. if not n_clusts_type else type_loss/n_clusts_type,
            })
        if compute_momentum:
            result.update({
                'p_accuracy': 0. if not n_clusts_momentum else p_acc/n_clusts_momentum,
                'p_loss': 0. if not n_clusts_momentum else p_loss/n_clusts_momentum,
            })
        if compute_vtx:
            result.update({
                'vtx_score_loss': 0. if not n_clusts_vtx else vtx_score_loss/n_clusts_vtx,
                'vtx_score_acc': 0. if not n_clusts_vtx else vtx_score_acc/n_clusts_vtx,
                'vtx_position_loss': 0. if not n_clusts_vtx_positives else vtx_position_loss/n_clusts_vtx_positives,
                'vtx_position_acc': 0. if not n_clusts_vtx_positives else vtx_position_acc/n_clusts_vtx_positives,
            })

        return result