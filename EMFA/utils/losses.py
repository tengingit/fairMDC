import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDimPrototypeLoss(nn.Module):
    def __init__(self, num_dim, num_per_dim, dim_z, momentum=0.9, warmup_epochs=100):
        super().__init__()
        self.num_dim = num_dim
        self.max_c = max(num_per_dim)
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        
        # 基础缓冲区 (EMA 存储)
        self.register_buffer("base_proto", torch.zeros(num_dim, self.max_c, dim_z))
        self.register_buffer("proto_init", torch.zeros(num_dim, self.max_c, dtype=torch.bool))
        
        # 统计先验缓冲区 (由外部 compute_all_priors 计算后加载)
        self.register_buffer("prior_pi", torch.zeros(num_dim, self.max_c))
        self.register_buffer("alpha", torch.zeros(num_dim, self.max_c))
        self.register_buffer("weight_k", torch.zeros(num_dim, self.max_c, num_dim))

    @torch.no_grad()
    def update_base_proto(self, z_list, labels):
        """
        z_list: 模型输出的特征列表 [z_1, z_2, ..., z_q]
        labels: [B, q]
        """
        for j in range(self.num_dim):
            z = z_list[j]
            y = labels[:, j]
            unique_labels = torch.unique(y)
            for c in unique_labels:
                if c < 0: continue
                mask = (y == c)
                z_mean = z[mask].mean(dim=0)
                
                if not self.proto_init[j, c]:
                    self.base_proto[j, c] = z_mean
                    self.proto_init[j, c] = True
                else:
                    self.base_proto[j, c] = self.momentum * self.base_proto[j, c] + \
                                           (1 - self.momentum) * z_mean

    @torch.no_grad()
    def build_corrected_proto(self):
        """
        基于凸组合 alpha 和 压缩率 k 构建修正原型
        """
        # 1. 计算各维度全局期望 E: [q, dim_z]
        init_mask = self.proto_init.unsqueeze(-1) # [q, max_c, 1]
        E = (self.prior_pi.unsqueeze(-1) * self.base_proto * init_mask).sum(dim=1)
        
        # 2. 计算跨维度补偿项: [q, max_c, dim_z]
        # 直接使用 k 作为权重 (相关性越高，贡献越大)
        # compensation[j1, c] = sum_{j2 != j1} k[j1, c, j2] * E[j2]
        compensation = torch.matmul(self.weight_k.view(-1, self.num_dim), E)
        compensation = compensation.view(self.num_dim, self.max_c, -1)
        compensation = compensation / (self.num_dim - 1)
        
        # 3. 凸组合逻辑: (1-a)*p + a*compensation
        a = self.alpha.unsqueeze(-1) # [q, max_c, 1]
        corrected_proto = (1 - a) * self.base_proto + a * compensation
        
        return corrected_proto

    def forward(self, z_list, labels, epoch):
        """
        对应你 train 函数中的 loss_proto = proto_loss(Z_list, Y, epoch)
        """
        # EMA 更新始终进行，以保持原型新鲜度
        self.update_base_proto(z_list, labels)

        # Warmup 期间不计算 Proto Loss，防止不稳定的原型干扰训练
        if epoch < self.warmup_epochs:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        # 构建修正后的公平性目标原型
        corrected_p = self.build_corrected_proto()
        
        proto_loss_sum = 0.0
        sample_count = 0
        
        for j in range(self.num_dim):
            y = labels[:, j]
            z = z_list[j]
            
            # 过滤：必须是有效标签且原型已初始化
            valid_mask = (y >= 0) & self.proto_init[j, y]
            if not valid_mask.any(): continue
            
            z_valid = z[valid_mask]
            p_target = corrected_p[j, y[valid_mask]]
            
            # 使用余弦距离解决数值爆炸问题
            sim = F.cosine_similarity(z_valid, p_target, dim=-1)
            proto_loss_sum += (1.0 - sim).sum()
            sample_count += z_valid.size(0)
            
        return proto_loss_sum / (sample_count + 1e-8)