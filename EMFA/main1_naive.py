# import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import math
from collections import defaultdict
from dataloader import *
from utils.utils import Init_random_seed
from model1 import FairMDC
# from model import MyImagenet
from config import *
from utils.metrics import eva, worst_dim_eva, worst_class_eva, variance_eva
from utils.losses import CrossDimPrototypeLoss

parser = argparse.ArgumentParser()
parser.add_argument('-dataset','--dataset', type=str, default="Flare1", help='dataset on which the experiment is conducted')
parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size for one iteration during training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1, help='learning rate parameter')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay parameter')
parser.add_argument('-max_epoch', '--max_epoch', type=int, default=500, help='maximal training epochs')
parser.add_argument('-wm_epoch', '--warmup_epoch', type=int, default=100, help='warm-up training epochs')
parser.add_argument('-lambda', '--lambda_tradeoff', type=float, default=1.0, help='trade-off parameter for proto loss')
parser.add_argument('-momentum', '--momentum', type=float, default=0.1, help='momentum parameter for prototype update')
parser.add_argument('-tau', '--tau', type=float, default=1.0, help='temperature parameter for logsumexp')
parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='the dimensionality of hidden layers.')
parser.add_argument('-dz', '--dim_z', type=int, default=512, help='the dimensionality of latent variable Z.')
parser.add_argument('-fd', '--fair_dimension', action='store_true', help='whether to study dimension-wise fairness')
parser.add_argument('-fl', '--fair_label', action='store_true', help='whether to study label-wise fairness')
parser.add_argument('-cuda', '--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('-test', '--test_mode', action='store_true', help='whether to use existing model for testing only')
parser.add_argument('--default_cfg', '-default_cfg', action='store_true', help='whether to run experiment with default hyperparameters')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args,nfold):
    Init_random_seed(seed=0)
    # cudnn.deterministic = True

    dataset_name = args.dataset
    print(dataset_name)
    dataset = eval(dataset_name)()      #dataset = BeLaE()
    X, Y = dataset.get_data()
    # Y = dataset.get_data()
    # num_training = Y.size(0)            #number of training examples
    num_dim = Y.size(1)                 #number of dimensions(class variables)
    label_per_dim = {}                  #class labels in each dimension
    num_per_dim = torch.zeros((num_dim),dtype = int)  #number of class labels in each dimension
    for dim in range(num_dim):
        labelset = torch.unique(Y[:,dim])
        label_per_dim[dim] = list(labelset)
        num_per_dim[dim] = len(label_per_dim[dim])

    X, Y = X.to(device), Y.to(device)
    # Y = Y.to(device)

    configs = generate_default_config()
    configs['dataset'] = dataset
    configs['num_feature'] = X.size(1)
    configs['device'] = device
    configs['lambda'] = args.lambda_tradeoff
    configs['weight_decay'] = args.weight_decay
    configs['num_dim'] = num_dim
    configs['label_per_dim'] = label_per_dim
    configs['num_per_dim'] = num_per_dim
    configs['lr'] = args.learning_rate
    configs['warmup_epoch'] = args.warmup_epoch
    configs['max_epoch'] = args.max_epoch
    configs['hidden_size'] = args.hidden_size
    configs['dim_z'] = args.dim_z
    configs['momentum'] = args.momentum
    configs['fair_dimension'] = args.fair_dimension
    configs['fair_label'] = args.fair_label
    configs['tau'] = args.tau
    # Loading dataset-specific configs
    if args.default_cfg:
        eval('{}_configs'.format(dataset_name))(configs)
    # print(configs)
    # criterion_cls = torch.nn.CrossEntropyLoss()

    # set optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=configs['weight_decay'])
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=configs['weight_decay'])

    results = np.zeros((3,10))
    for fold in range(0,nfold):
        # run_id = time.strftime("%Y%m%d-%H%M%S")
        # runs_dir = f"runs1_n/{args.dataset}/fold{fold}/{run_id}"
        # writer = SummaryWriter(log_dir=runs_dir)
        best_ham = 0
        train_idx, test_idx = dataset.idx_cv(fold)
        # X_train = X[train_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]
        prior_pi, alpha, weight_k = compute_all_priors(Y_train, num_per_dim)
        proto_loss = CrossDimPrototypeLoss(
            num_dim=num_dim, 
            num_per_dim=num_per_dim, 
            dim_z=configs['dim_z'],
            momentum=args.momentum,
            warmup_epochs=args.warmup_epoch,
        )
        proto_loss.load_state_dict({
            "prior_pi": prior_pi,
            "alpha": alpha,
            "weight_k": weight_k
            }, strict=False)
        proto_loss = proto_loss.to(device)
        model = FairMDC(configs)
        # print(model)
        model = model.to(device)
        model.reset_parameters()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=configs['weight_decay']
        )

        # def lr_lambda(epoch):
        #     if epoch < args.warmup_epoch:
        #         return 1.0
        #     else:
        #         return 0.5 * (
        #             1 + math.cos(
        #                 math.pi * (epoch - args.warmup_epoch) / (args.max_epoch - args.warmup_epoch)
        #             )
        #         )

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_lambda
        # )

        # optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr':configs['lr'], 'momentum':0.9, 'weight_decay': configs['weight_decay']},
        #                         {'params': model.adapters.parameters(), 'lr':configs['lr'], 'momentum':0.9, 'weight_decay': configs['weight_decay']},
        #                         {'params': model.heads.parameters(), 'lr':configs['lr'], 'momentum':0.9, 'weight_decay': configs['weight_decay']}
        #                         ])

        file_saver = "/{ds}_lr{lr}_hs{hs}_dz{dz}_wd{wd}".format(ds=args.dataset,
                                                                hs=args.hidden_size,
                                                                dz=args.dim_z,
                                                                lr=args.learning_rate,
                                                                wd=args.weight_decay)
        print(file_saver)

        log_loss_path = "logs1_n/"+args.dataset+"/loss/fold"+str(fold)
        log_metric_path = "logs1_n/"+args.dataset+"/metric/fold"+str(fold)
        checkpoint_path = "checkpoints1_n/"+args.dataset+"/fold"+str(fold)
        result_loss_path = "results1_n/"+args.dataset+"/loss/fold"+str(fold)
        result_metric_path = "results1_n/"+args.dataset+"/metric/fold"+str(fold)
        result_matrix_path = "results1_n/"+args.dataset+"/matrix/fold"+str(fold)
        result_path = "results1_n/"+args.dataset
        time_path = "results1_n/time_500epochs"
        path_list = [log_loss_path, log_metric_path, checkpoint_path, result_loss_path, result_metric_path, result_matrix_path,time_path]
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)

        log_loss_table = np.zeros(shape=(args.max_epoch, 3))
        log_metric_table = np.zeros(shape=(args.max_epoch, 3))
        results_loss_table = np.zeros(shape=(args.max_epoch, 3))
        results_metric_table = np.zeros(shape=(args.max_epoch, 3))
        epoch_records = defaultdict(dict)

        train_loader, test_loader = data_loader(dataset, fold, batch_size=args.batch_size, shuffle=False)
        model_path = checkpoint_path + file_saver +".pth"
        if args.test_mode and os.path.exists(model_path):
            print('Loading existing models O.o')
            checkpoint = torch.load(model_path,weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = args.max_epoch
            test_loss, loss_cls_item, loss_proto_item, pred_Y_test = predict(epoch, test_loader, model, configs, proto_loss)
            test_ham, test_exa, test_sub  = eva(Y_test, pred_Y_test)
            test_ham, test_exa, test_sub = test_ham.cpu().numpy(), test_exa.cpu().numpy(), test_sub.cpu().numpy()
            worst_dim_accuracy, worst_dim_balanced_accuracy, worst_dim_f1_score = worst_dim_eva(Y_test, pred_Y_test)
            worst_class_accuracy_list, worst_class_macro_f1_score_list = worst_class_eva(Y_test, pred_Y_test)
            variance_list = variance_eva(Y_test, pred_Y_test)
            epoch_records[epoch]["ham"] = test_ham
            epoch_records[epoch]["exa"] = test_exa
            epoch_records[epoch]["sub"] = test_sub
            epoch_records[epoch]["worst_dim_accuracy"] = worst_dim_accuracy
            epoch_records[epoch]["worst_balanced_accuracy"] = worst_dim_balanced_accuracy
            epoch_records[epoch]["worst_f1_score"] = worst_dim_f1_score
            for j in range(num_dim):
                epoch_records[epoch][f"worst_class_accuracy/dim_{j}"] = worst_class_accuracy_list[j]
            for j in range(num_dim):
                epoch_records[epoch][f"worst_class_f1/dim_{j}"] = worst_class_macro_f1_score_list[j]
            for j in range(num_dim):
                epoch_records[epoch][f"variance/dim_{j}"] = variance_list[j]

        else:
            print('Fold'+str(fold)+': start training!')
            best_ham, best_epoch = 0, 0
            for epoch in range(args.max_epoch):
                train_loss, loss_cls_item, loss_proto_item, pred_Y_train = train(epoch, train_loader, model, optimizer, configs, proto_loss)
                # scheduler.step()
                
                train_ham, train_exa, train_sub = eva(Y_train, pred_Y_train)
                log_loss_table[epoch, :] = train_loss, loss_cls_item, loss_proto_item
                log_metric_table[epoch, :] = train_ham.cpu(), train_exa.cpu(), train_sub.cpu()
                np.savetxt(log_loss_path+file_saver+".csv", log_loss_table, delimiter=',', fmt='%1.4f')
                np.savetxt(log_metric_path+file_saver+".csv", log_metric_table, delimiter=',', fmt='%1.4f')

                test_loss, loss_cls_item, loss_proto_item, pred_Y_test = predict(epoch, test_loader, model, configs, proto_loss)
                test_ham, test_exa, test_sub  = eva(Y_test, pred_Y_test)
                test_ham, test_exa, test_sub  = test_ham.cpu().numpy(), test_exa.cpu().numpy(), test_sub.cpu().numpy()
                worst_dim_accuracy, worst_dim_balanced_accuracy, worst_dim_f1_score = worst_dim_eva(Y_test, pred_Y_test)
                worst_class_accuracy_list, worst_class_macro_f1_score_list = worst_class_eva(Y_test, pred_Y_test)
                variance_list = variance_eva(Y_test, pred_Y_test)

                results_loss_table[epoch, :] = test_loss, loss_cls_item, loss_proto_item
                results_metric_table[epoch, :] = test_ham, test_exa, test_sub
                np.savetxt(result_loss_path+file_saver+".csv", results_loss_table, delimiter=',', fmt='%1.4f')
                np.savetxt(result_metric_path+file_saver+".csv", results_metric_table, delimiter=',', fmt='%1.4f')
                if (epoch+1) % 50 == 0:
                    print('[{}/{}] Training :'.format(epoch + 1, args.max_epoch))
                    print(f"train_loss:{train_loss},\ntrain_ham:{train_ham}, train_exa:{train_exa}, train_sub:{train_sub}")
                    print(f"test_loss:{test_loss},\ntest_ham:{test_ham}, test_exa:{test_exa}, test_sub:{test_sub}")
                    print(f"worst_dim_accuracy:{worst_dim_accuracy}, worst_dim_balanced_accuracy:{worst_dim_balanced_accuracy}")
                    print(f"worst_dim_f1_score:{worst_dim_f1_score}")
                    for j in range(num_dim):
                        print(f"worst_class_accuracy/dim_{j}: {worst_class_accuracy_list[j]}, worst_class_f1/dim_{j}: {worst_class_macro_f1_score_list[j]}")
                        print(f"variance/dim_{j}: {variance_list[j]}")

                if test_ham > best_ham:
                    best_ham = test_ham
                    best_epoch = epoch
                
                # writer.add_scalar("acc/ham", test_ham, epoch)
                # writer.add_scalar("acc/exa", test_exa, epoch)
                # writer.add_scalar("acc/sub", test_sub, epoch)
                # writer.add_scalar("acc/worst_accuracy", worst_dim_accuracy, epoch)
                # writer.add_scalar("recall/worst_balanced_accuracy", worst_dim_balanced_accuracy, epoch)
                # writer.add_scalar("f1/worst_f1_score", worst_dim_f1_score, epoch)
                # for j in range(num_dim):
                #     writer.add_scalar(f"recall/worst_class/dim_{j}", worst_class_accuracy_list[j], epoch)
                #     writer.add_scalar(f"f1/worst_class/dim_{j}", worst_class_macro_f1_score_list[j], epoch)
                #     writer.add_scalar(f"variance/dim_{j}", variance_list[j], epoch)
                # writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

                epoch_records[epoch]["ham"] = test_ham
                epoch_records[epoch]["exa"] = test_exa
                epoch_records[epoch]["sub"] = test_sub
                epoch_records[epoch]["worst_dim_accuracy"] = worst_dim_accuracy
                epoch_records[epoch]["worst_balanced_accuracy"] = worst_dim_balanced_accuracy
                epoch_records[epoch]["worst_f1_score"] = worst_dim_f1_score
                for j in range(num_dim):
                    epoch_records[epoch][f"worst_class_accuracy/dim_{j}"] = worst_class_accuracy_list[j]
                for j in range(num_dim):
                    epoch_records[epoch][f"worst_class_f1/dim_{j}"] = worst_class_macro_f1_score_list[j]
                for j in range(num_dim):
                    epoch_records[epoch][f"variance/dim_{j}"] = variance_list[j]

            # save model of the last epoch
            torch.save({'best_epoch': best_epoch+1, 'best_ham': best_ham, 
                        'state_dict': model.state_dict()}, model_path)
            
        results[:,fold] = test_ham, test_exa, test_sub
        df_pred = pd.DataFrame(pred_Y_test.cpu().numpy())
        csv_path = result_matrix_path + file_saver + f".csv"
        df_pred.to_csv(csv_path, index=False, header=False)

        df = (
            pd.DataFrame.from_dict(epoch_records, orient="index")
            .sort_index()
            .reset_index()
            .rename(columns={"index": "epoch"})
        )
        df.to_csv(result_metric_path + file_saver + "_metrics.csv", index=False, float_format="%.4f")
                
    df = pd.DataFrame(results,index=['hammingscore','exactmatch','subexactmatch'])
    df = df.T
    df.to_csv(result_path+file_saver+".csv")

def train(epoch, train_loader, model, optimizer, configs, proto_loss):
    model.train()
    train_loss = 0
    loss_cls_item, loss_proto_item = 0, 0
    pred_Y = []
    for batch_idx, (X, Y) in enumerate(train_loader):
        pred_Y_batch = []
        X, Y = X.to(device), Y.to(device)
        num_dim = Y.size(1)
        Z_list, logits = model(X)
        loss_clses = []
        # loss_cls = torch.tensor(0, dtype=torch.float32).to(device)
        for dim, logit in enumerate(logits):
            loss_clses.append(F.cross_entropy(logit,Y[:,dim]))
            pred_Y_batch.append(torch.argmax(logit, dim=1, keepdim=True))                  #(b,1)
        pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
        pred_Y.append(pred_Y_batch)
        loss_clses = torch.stack(loss_clses)

        if configs['fair_dimension']:
            loss_cls = torch.logsumexp(configs['tau'] * loss_clses, dim=0) / configs['tau']  
        else:
            loss_cls = loss_clses.mean()

        loss_proto = torch.tensor(0.0, device=device)
        if configs['fair_label']:
            loss_proto = proto_loss(Z_list, Y, epoch)
            
        loss = loss_cls + configs['lambda'] * loss_proto
        optimizer.zero_grad()    
        loss.backward()          
        optimizer.step()         
        train_loss += loss.item()
        loss_cls_item += loss_cls.item()
        loss_proto_item += loss_proto.item()

        global_step = epoch * len(train_loader) + batch_idx
        # writer.add_scalar("loss/total", loss.item(), global_step)
        # writer.add_scalar("loss/ce", loss_cls.item(), global_step)
        # writer.add_scalar("loss/proto", loss_proto.item(), global_step)

        # writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)


    pred_Y = torch.cat(pred_Y, dim=0)

    return train_loss, loss_cls_item, loss_proto_item, pred_Y

def predict(epoch, test_loader, model, configs, proto_loss):
    pred_Y = []
    test_loss = 0
    loss_cls_item, loss_proto_item = 0, 0
    with torch.no_grad():
        model.eval()
        for X, Y in test_loader:
            pred_Y_batch = []
            X, Y = X.to(device), Y.to(device)
            num_dim = Y.size(1)
            Z_list, logits = model(X)
            loss_clses = []
            # loss_cls = torch.tensor(0, dtype=torch.float32).to(device)
            for dim, logit in enumerate(logits):
                loss_clses.append(F.cross_entropy(logit,Y[:,dim]))
                pred_Y_batch.append(torch.argmax(logit, dim=1, keepdim=True))                  #(b,1)
            pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
            pred_Y.append(pred_Y_batch)
            loss_clses = torch.stack(loss_clses)

            if configs['fair_dimension']:
                loss_cls = torch.logsumexp(configs['tau'] * loss_clses, dim=0) / configs['tau']  
            else:
                loss_cls = loss_clses.mean()

            loss_proto = torch.tensor(0.0, device=device)
            if configs['fair_label']:
                loss_proto = proto_loss(Z_list, Y, epoch)

            loss = loss_cls + configs['lambda'] * loss_proto
            test_loss += loss.item()      
            loss_cls_item += loss_cls.item()
        loss_proto_item += loss_proto.item()  
    
        pred_Y = torch.cat(pred_Y, dim=0)

    return test_loss, loss_cls_item, loss_proto_item, pred_Y


def compute_all_priors(Y_train, num_per_dim):
    num_training, num_dim = Y_train.shape
    max_c = max(num_per_dim)
    
    prior_pi = torch.zeros(num_dim, max_c)
    alpha = torch.zeros(num_dim, max_c)
    weight_k = torch.zeros(num_dim, max_c, num_dim)

    for j1 in range(num_dim):
        K1 = num_per_dim[j1]
        counts = torch.bincount(Y_train[:, j1], minlength=K1).float()
        
        # 先验 pi
        prior_pi[j1, :K1] = counts / num_training
        # 置信度 alpha (平方根反比)
        alpha[j1, :K1] = 1.0 / (torch.sqrt(counts) + 1.0)
        
        for c in range(K1):
            mask = (Y_train[:, j1] == c)
            if mask.sum() == 0: continue
            
            for j2 in range(num_dim):
                if j2 == j1: continue
                unique_labels_j2 = torch.unique(Y_train[mask, j2])
                # 新定义：压缩率/聚焦度 k
                weight_k[j1, c, j2] = 1.0 - (len(unique_labels_j2) / num_per_dim[j2])

    return prior_pi, alpha, weight_k


def compute_cross_diversity(Y_train, num_per_dim):
    """
    Returns:
        cross_diversity[j1][c][j2] = k_{c,j1}^{(j2)}
    """
    num_dim = Y_train.size(1)
    cross_div = [
        [ [0 for _ in range(num_dim)] for _ in range(num_per_dim[j1]) ]
        for j1 in range(num_dim)
    ]

    for j1 in range(num_dim):
        for c in range(num_per_dim[j1]):
            mask = (Y_train[:, j1] == c)
            if mask.sum() == 0:
                continue

            for j2 in range(num_dim):
                if j2 == j1:
                    continue

                labels_j2 = torch.unique(Y_train[mask, j2])
                cross_div[j1][c][j2] = num_per_dim[j2]/len(labels_j2)

    return cross_div


if __name__ == '__main__':
    args = parser.parse_args()
    namesets = ['Adult','BeLaE','CoIL2000','Default','Disfa','Edm', 
                'Enb','Fera','Flare1','Flickr','Jura','Pain','Rf1',
                'Song','Thyroid','TIC2000','Voice']
    lr_list = [1e-1,1e-2]

    time_dict = {}
    start_time = time.time()
    temp_time0 = time.time()
    for name in namesets:
        args.dataset = name
        for lr in lr_list:
            args.learning_rate = lr
            main(args,nfold=10)
            temp_time1 = time.time()
            time_dict[name] = temp_time1 - temp_time0
            file_saver = "/{ds}_lr{lr}_hs{hs}_dz{dz}_wd{wd}".format(ds=args.dataset,
                                                                    hs=args.hidden_size,
                                                                    dz=args.dim_z,
                                                                    lr=args.learning_rate,
                                                                    wd=args.weight_decay)
            time_path = "results1_n/time_500epochs" + file_saver + ".txt"
            with open(time_path, "a") as f:
                f.write(name + ":" + str(time_dict[name]) + 's' + '\n')

        file_saver = "/lr{lr}_hs{hs}_dz{dz}_wd{wd}".format(hs=args.hidden_size,
                                                           dz=args.dim_z,
                                                           lr=args.learning_rate,
                                                           wd=args.weight_decay)
        
        time_path = "results1_n/time_500epochs" + file_saver + ".txt"
        with open(time_path, "a") as f:
            for k, v in time_dict.items():
                f.write(k + ':' + str(v) + 's' + '\n')
                
    end_time = time.time()    
    print("during {:.2f}s".format(end_time - start_time))

