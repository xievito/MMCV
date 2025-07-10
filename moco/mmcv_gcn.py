import torch
import torch.nn as nn
import torch.nn.functional as F

from .AGCN import Model as AGCN


def multi_nce_loss(logits, mask):
    mask_sum = mask.sum(1)
    loss = - torch.log((F.softmax(logits, dim=1) * mask).sum(1) / mask_sum)
    return loss.mean()

class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, TT, TS):
        super(KLDiv, self).__init__()
        self.TT = TT
        self.TS = TS

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.TS, dim=1)
        p_t = F.softmax(y_t/self.TT, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean')
        return loss

class MoCo(nn.Module):
    def __init__(self, skeleton_representation, args_agcn, dim=128, K=16384, T=0.07,
                 teacher_T=0.05, student_T=0.1, alpha=1.0, beta=0.25, topk=1024, mk=0.999, mp=0.995, mlp=False, pretrain=True):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = AGCN(**args_agcn)
            self.encoder_q_motion = AGCN(**args_agcn)
            self.encoder_q_bone = AGCN(**args_agcn)
        else:
            self.K = K
            self.T = T
            self.mk = mk
            self.mp = mp
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.alpha = alpha
            self.beta = beta
            self.kl_loss = KLDiv(self.teacher_T, self.student_T)
            self.topk = topk
            mlp = mlp
            print(" MoCo parameters", K, mk, mp, T, mlp)
            print(" MMCV parameters: teacher-T %.2f, student-T %.2f, alpha: %.2f, beta: %.2f, topk: %d" % (
            teacher_T, student_T, alpha, beta, topk))
            print(skeleton_representation)

            self.encoder_q = AGCN(**args_agcn)
            self.encoder_k = AGCN(**args_agcn)
            self.encoder_p = AGCN(**args_agcn)
            self.encoder_q_motion = AGCN(**args_agcn)
            self.encoder_k_motion = AGCN(**args_agcn)
            self.encoder_p_motion = AGCN(**args_agcn)
            self.encoder_q_bone = AGCN(**args_agcn)
            self.encoder_k_bone = AGCN(**args_agcn)
            self.encoder_p_bone = AGCN(**args_agcn)

            # projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
                self.encoder_p.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                                         self.encoder_q_motion.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                                         self.encoder_k_motion.fc)
                self.encoder_p_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                                         self.encoder_p_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q_bone.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k_bone.fc)
                self.encoder_p_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p_bone.fc)

            for param_q, param_k, param_p in zip(self.encoder_q.parameters(), self.encoder_k.parameters(),
                                                 self.encoder_p.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
                param_p.data.copy_(param_q.data)
                param_p.requires_grad = False
            for param_q, param_k, param_p in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters(),
                                                 self.encoder_p_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
                param_p.data.copy_(param_q.data)
                param_p.requires_grad = False
            for param_q, param_k, param_p in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters(),
                                                 self.encoder_p_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
                param_p.data.copy_(param_q.data)
                param_p.requires_grad = False

            # create the queue
            self.register_buffer("queue_k", torch.randn(dim, self.K))
            self.queue_k = F.normalize(self.queue_k, dim=0)
            self.register_buffer("queue_ptr_k", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_p", torch.randn(dim, self.K))
            self.queue_p = F.normalize(self.queue_p, dim=0)
            self.register_buffer("queue_ptr_p", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion_k", torch.randn(dim, self.K))
            self.queue_motion_k = F.normalize(self.queue_motion_k, dim=0)
            self.register_buffer("queue_ptr_motion_k", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_motion_p", torch.randn(dim, self.K))
            self.queue_motion_p = F.normalize(self.queue_motion_p, dim=0)
            self.register_buffer("queue_ptr_motion_p", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone_k", torch.randn(dim, self.K))
            self.queue_bone_k = F.normalize(self.queue_bone_k, dim=0)
            self.register_buffer("queue_ptr_bone_k", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_bone_p", torch.randn(dim, self.K))
            self.queue_bone_p = F.normalize(self.queue_bone_p, dim=0)
            self.register_buffer("queue_ptr_bone_p", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k, param_p in zip(self.encoder_q.parameters(), self.encoder_k.parameters(),
                                             self.encoder_p.parameters()):
            param_k.data = param_k.data * self.mk + param_q.data * (1. - self.mk)
            param_p.data = param_p.data * self.mp + param_q.data * (1. - self.mp)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k, param_p in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters(),
                                             self.encoder_p_motion.parameters()):
            param_k.data = param_k.data * self.mk + param_q.data * (1. - self.mk)
            param_p.data = param_p.data * self.mp + param_q.data * (1. - self.mp)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k, param_p in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters(),
                                             self.encoder_p_bone.parameters()):
            param_k.data = param_k.data * self.mk + param_q.data * (1. - self.mk)
            param_p.data = param_p.data * self.mp + param_q.data * (1. - self.mp)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, p):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_k)
        self.queue_k[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_k[0] = ptr

        batch_size = p.shape[0]
        ptr = int(self.queue_ptr_p)
        self.queue_p[:, ptr:ptr + batch_size] = p.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr_p[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys, p):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion_k)
        self.queue_motion_k[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_motion_k[0] = ptr

        batch_size = p.shape[0]
        ptr = int(self.queue_ptr_motion_p)
        self.queue_motion_p[:, ptr:ptr + batch_size] = p.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr_motion_p[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys, p):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone_k)
        self.queue_bone_k[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_bone_k[0] = ptr

        batch_size = p.shape[0]
        ptr = int(self.queue_ptr_bone_p)
        self.queue_bone_p[:, ptr:ptr + batch_size] = p.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_bone_p[0] = ptr

    @torch.no_grad()
    def cal_motion(self, im):
        im_motion = torch.zeros_like(im)
        im_motion[:, :, :-1, :, :] = im[:, :, 1:, :, :] - im[:, :, :-1, :, :]
        return im_motion

    @torch.no_grad()
    def cal_bone(self, im):
        im_bone = torch.zeros_like(im)
        for v1, v2 in self.Bone:
            im_bone[:, :, :, v1 - 1, :] = im[:, :, :, v1 - 1, :] - im[:, :, :, v2 - 1, :]
        return im_bone

    @torch.no_grad()
    def reshape_im(self, im):
        im_q_motion = self.cal_motion(im)
        im_q_bone = self.cal_bone(im)

        return im, im_q_motion, im_q_bone

    def forward(self, im_q, im_k=None, im_p=None, im_r=None, view='joint', knn_eval=False):
        im_q, im_q_motion, im_q_bone = self.reshape_im(im_q)
        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion, knn_eval)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone, knn_eval)
            elif view == 'all':
                return (self.encoder_q(im_q, knn_eval) + \
                        self.encoder_q_motion(im_q_motion, knn_eval) + \
                        self.encoder_q_bone(im_q_bone, knn_eval)) / 3.
            else:
                raise ValueError

        im_k, im_k_motion, im_k_bone = self.reshape_im(im_k)
        im_p, im_p_motion, im_p_bone = self.reshape_im(im_p)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        # compute key features for  s1 and  s2  skeleton representations
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)
            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

            p = self.encoder_p(im_p)
            p = F.normalize(p, dim=1)
            p_motion = self.encoder_p_motion(im_p_motion)
            p_motion = F.normalize(p_motion, dim=1)
            p_bone = self.encoder_p_bone(im_p_bone)
            p_bone = F.normalize(p_bone, dim=1)

        # MOCO
        l_pos_qk = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg_qk = torch.einsum('nc,ck->nk', [q, self.queue_k.clone().detach()])
        l_pos_motion_qk = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion_qk = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion_k.clone().detach()])
        l_pos_bone_qk = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone_qk = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone_k.clone().detach()])

        l_pos_qp = torch.einsum('nc,nc->n', [q, p]).unsqueeze(-1)
        l_neg_qp = torch.einsum('nc,ck->nk', [q, self.queue_p.clone().detach()])
        l_pos_motion_qp = torch.einsum('nc,nc->n', [q_motion, p_motion]).unsqueeze(-1)
        l_neg_motion_qp = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion_p.clone().detach()])
        l_pos_bone_qp = torch.einsum('nc,nc->n', [q_bone, p_bone]).unsqueeze(-1)
        l_neg_bone_qp = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone_p.clone().detach()])

        logits_qpk = torch.cat([l_pos_qp, l_neg_qp, l_neg_qk], dim=1) / self.T
        logits_motion_qpk = torch.cat([l_pos_motion_qp, l_neg_motion_qp, l_neg_motion_qk], dim=1) / self.T
        logits_bone_qpk = torch.cat([l_pos_bone_qp, l_neg_bone_qp, l_neg_bone_qk], dim=1) / self.T

        logits_qkp = torch.cat([l_pos_qk, l_neg_qk, l_neg_qp], dim=1) / self.T
        logits_motion_qkp = torch.cat([l_pos_motion_qk, l_neg_motion_qk, l_neg_motion_qp], dim=1) / self.T
        logits_bone_qkp = torch.cat([l_pos_bone_qk, l_neg_bone_qk, l_neg_bone_qp], dim=1) / self.T

        logits_jm = torch.cat([l_pos_qk, l_neg_qk, l_neg_motion_qk], dim=1) / self.T
        logits_mj = torch.cat([l_pos_motion_qk, l_neg_motion_qk, l_neg_qk], dim=1) / self.T
        logits_jb = torch.cat([l_pos_qk, l_neg_qk, l_neg_bone_qk], dim=1) / self.T
        logits_bj = torch.cat([l_pos_bone_qk, l_neg_bone_qk, l_neg_qk], dim=1) / self.T
        logits_mb = torch.cat([l_pos_motion_qk, l_neg_motion_qk, l_neg_bone_qk], dim=1) / self.T
        logits_bm = torch.cat([l_pos_bone_qk, l_neg_bone_qk, l_neg_motion_qk], dim=1) / self.T

        topk_onehot = torch.zeros_like(l_neg_qk)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot, topk_onehot], dim=1)

        l_info_intra = multi_nce_loss(logits_qpk, pos_mask) + multi_nce_loss(logits_qkp, pos_mask) + \
                   multi_nce_loss(logits_bone_qpk, pos_mask) + multi_nce_loss(logits_bone_qkp, pos_mask) + \
                   multi_nce_loss(logits_motion_qpk, pos_mask) + multi_nce_loss(logits_motion_qkp, pos_mask)
        l_info_intra /= 6

        l_info_inter = multi_nce_loss(logits_jm, pos_mask) + multi_nce_loss(logits_mj, pos_mask) + \
                  multi_nce_loss(logits_jb, pos_mask) + multi_nce_loss(logits_bj, pos_mask) + \
                  multi_nce_loss(logits_mb, pos_mask) + multi_nce_loss(logits_bm, pos_mask)
        l_info_inter /= 6

        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue_k.clone().detach()])
        lk_neg_motion = torch.einsum('nc,ck->nk', [k_motion, self.queue_motion_k.clone().detach()])
        lk_neg_bone = torch.einsum('nc,ck->nk', [k_bone, self.queue_bone_k.clone().detach()])
        # Top-k
        lk_neg_topk, k_topk_idx = torch.topk(lk_neg, self.topk, dim=-1)
        lk_neg_motion_topk, k_motion_topk_idx = torch.topk(lk_neg_motion, self.topk, dim=-1)
        lk_neg_bone_topk, k_bone_topk_idx = torch.topk(lk_neg_bone, self.topk, dim=-1)

        lp_neg = torch.einsum('nc,ck->nk', [p, self.queue_p.clone().detach()])
        lp_neg_motion = torch.einsum('nc,ck->nk', [p_motion, self.queue_motion_p.clone().detach()])
        lp_neg_bone = torch.einsum('nc,ck->nk', [p_bone, self.queue_bone_p.clone().detach()])

        lp_neg_topk, p_topk_idx = torch.topk(lp_neg, self.topk, dim=-1)
        lp_neg_motion_topk, p_motion_topk_idx = torch.topk(lp_neg_motion, self.topk, dim=-1)
        lp_neg_bone_topk, p_bone_topk_idx = torch.topk(lp_neg_bone, self.topk, dim=-1)

        l_kd_intra = self.kl_loss(torch.gather(l_neg_qk, -1, p_topk_idx), lp_neg_topk) + \
                    self.kl_loss(torch.gather(l_neg_qp, -1, k_topk_idx), lk_neg_topk) + \
                    self.kl_loss(torch.gather(l_neg_motion_qk, -1, p_motion_topk_idx), lp_neg_motion_topk) + \
                    self.kl_loss(torch.gather(l_neg_motion_qp, -1, k_motion_topk_idx), lk_neg_motion_topk) + \
                    self.kl_loss(torch.gather(l_neg_bone_qk, -1, p_bone_topk_idx), lp_neg_bone_topk) + \
                    self.kl_loss(torch.gather(l_neg_bone_qp, -1, k_bone_topk_idx), lk_neg_bone_topk)

        l_kd_inter = self.kl_loss(torch.gather(l_neg_motion_qk, -1, k_topk_idx), lk_neg_topk) + \
                     self.kl_loss(torch.gather(l_neg_bone_qk, -1, k_topk_idx), lk_neg_topk) + \
                     self.kl_loss(torch.gather(l_neg_motion_qp, -1, p_topk_idx), lp_neg_topk) + \
                     self.kl_loss(torch.gather(l_neg_bone_qp, -1, p_topk_idx), lp_neg_topk)+\
                     self.kl_loss(torch.gather(l_neg_qk, -1, k_motion_topk_idx), lk_neg_motion_topk) + \
                     self.kl_loss(torch.gather(l_neg_bone_qk, -1, k_motion_topk_idx), lk_neg_motion_topk) + \
                     self.kl_loss(torch.gather(l_neg_qp, -1, p_motion_topk_idx), lp_neg_motion_topk) + \
                     self.kl_loss(torch.gather(l_neg_bone_qp, -1, p_motion_topk_idx), lp_neg_motion_topk)+\
                     self.kl_loss(torch.gather(l_neg_qk, -1, k_bone_topk_idx), lk_neg_bone_topk) + \
                     self.kl_loss(torch.gather(l_neg_motion_qk, -1, k_bone_topk_idx), lk_neg_bone_topk) + \
                     self.kl_loss(torch.gather(l_neg_qp, -1, p_bone_topk_idx), lp_neg_bone_topk) + \
                     self.kl_loss(torch.gather(l_neg_motion_qp, -1, p_bone_topk_idx), lp_neg_bone_topk)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, p)
        self._dequeue_and_enqueue_motion(k_motion, p_motion)
        self._dequeue_and_enqueue_bone(k_bone, p_bone)

        return (l_info_intra + l_info_inter) * self.alpha, (l_kd_intra + l_kd_inter) * self.beta
