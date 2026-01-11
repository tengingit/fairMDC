import torch
import torch.nn as nn
import torchvision
from mlp import MLP, MLP_br
from utils.utils import Init_random_seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FairMDC(nn.Module):
    def __init__(self, configs):
        super(FairMDC, self).__init__()
        self.rand_seed = configs['rand_seed']
        self.num_dim = configs['num_dim']
        # self.hs = configs['hidden_size']
        # self.z_dim = configs['dim_z']
        # self.num_per_dim = configs['num_per_dim']

        # ===== g(x): shared backbone =====
        # self.backbone = MLP(
        #             configs['num_feature'],
        #             configs['dim_z'],
        #             [configs['hidden_size']],
        #             batchNorm=False,dropout=True,nonlinearity='relu',
        #             with_output_nonlinearity=True
        #         )
        
         # ===== g_j(x): dimension-specific adapters =====
        self.adapters = nn.ModuleList([
            MLP(
            configs['num_feature'],
            configs['dim_z'],
            [configs['hidden_size']],
            batchNorm=False,dropout=False,nonlinearity='relu',
            with_output_nonlinearity=True
            )
            for j in range(self.num_dim)
        ])

        # ===== h_j(x): heads =====
        self.heads = nn.ModuleList([
            MLP(
                configs['dim_z'],
                configs['num_per_dim'][j],
                [],
                batchNorm=False,dropout=False,nonlinearity='relu',
                with_output_nonlinearity=False
            )
            for j in range(self.num_dim)
        ])

        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        # self.backbone.reset_parameters()
        for adapter in self.adapters:
            adapter.reset_parameters()
        for head in self.heads:
            head.reset_parameters()

    def forward(self, X):
        # Z = self.backbone(X)
        Z_list, logits = [], []

        for j in range(self.num_dim):
            Z_j = self.adapters[j](X)
            logit_j = self.heads[j](Z_j)

            Z_list.append(Z_j)
            logits.append(logit_j)

        return Z_list, logits
    

    
    
# ResNet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# # ResNet = torchvision.models.resnet18(weights=None)
# # ResNet.eval()
# # ResNet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# backbone = list(ResNet.children())[:-1]    #去掉全连接层
# encode_Res = nn.Sequential(*backbone).to(device)

# class MyImagenet(FairMDC):
#     def __init__(self, configs, br=True, cp=True):
#         super(MyImagenet, self).__init__(configs, br=br, cp=cp)
#         self.backbone = encode_Res

#     def forward(self, X):
#         X = self.backbone(X).squeeze()
#         Z = self.encoder(X)
#         X_hat = self.decoder(Z).sigmoid_()
#         XZ = torch.cat((X,Z),dim=1)                        # (b, num_feature+dim_z)
#         pred_probs = []
#         if self.br:
#             for i in range(self.num_dim):
#                 output = self.classifiers[i](XZ)
#                 pred_probs.append(output)

#         if self.cp:
#             joint_probs = self.joint_classifier(XZ)
#             return Z, X_hat, pred_probs, joint_probs

#         return Z, X_hat, pred_probs

#     def get_embs_backbone(self, X):
#         embeddings_backbone = self.backbone(X).squeeze()
#         return embeddings_backbone

        

if __name__ == '__main__':
    # print(ResNet)
    # print("  ")
    # print(encode_Res)
    configs = {}
    configs['num_dim'] = 2
    configs['dim_z'] = 32
    configs['num_feature'] = 512
    configs['hidden_size'] = 256
    configs['num_per_dim'] = [2,3]
    configs['num_valid'] = 10
    # model = VarMDC(configs)
    # for name,para in model.named_parameters():
    #     # if "backbone" in name:
    #     #     para.requires_grad = False
    #     print(name)
    #     print(para)
    #     break
    # model = LatentVariableGenerator(dim_z=16)
    # # X = torch.ones((32,32))
    # for i in range(2):
    #     Z = model(4)
    #     print(Z)
    #     Z = model(2)
    #     print(Z)
    

