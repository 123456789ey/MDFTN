import os.path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from loader import *
from decode_onedown import MSDnet
from measure import compute_measure
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips

class Solver(object):
    def __init__(self, args, data_loader):
        self.n_d_train =args.n_d_train
        self.lambda_ = args.lambda_
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MSDnet()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.lr = args.lr
        #self.criterion = nn.MSELoss()  # 误差
        self.criterion = nn.L1Loss()
        #self.loss_fn = lpips.LPIPS(net='vgg',spatial='spatial',pnet_tune=True).to(self.device)  #,pnet_tune=True
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        # 对优化器的学习率进行调整，每经过step_size之后，学习率乘以gamma
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.decay_iters, gamma=0.5)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'model_{}iter.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'model_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(f))  # 将训练的模型参数加载到文件夹f中

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result, fname):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, fname, 'result_{}.png'.format(fig_name)))
        plt.close()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self):
        total_iter = 0
        pred_psnr_avg_source1, pred_ssim_avg_source1, pred_rmse_avg_source1 = 0, 0, 0
        pred_psnr_avg_source2, pred_ssim_avg_source2, pred_rmse_avg_source2 = 0, 0, 0
        pred_psnr_avg_source3, pred_ssim_avg_source3, pred_rmse_avg_source3 = 0, 0, 0
        param = self.get_parameter_number(self.model)
        print('param=', param)
        start_time = time.time()
        for epoch in range(0, self.num_epochs):
            self.model.train(True)
            for iter , ((x_source1, y_source1), (x_source2, y_source2), (x_source3, y_source3)) in enumerate(self.data_loader):
                total_iter += 1
                x_source1 = x_source1.unsqueeze(0).float().to(self.device)
                y_source1 = y_source1.unsqueeze(0).float().to(self.device)
                x_source2 = x_source2.unsqueeze(0).float().to(self.device)
                y_source2 = y_source2.unsqueeze(0).float().to(self.device)
                x_source3 = x_source3.unsqueeze(0).float().to(self.device)
                y_source3 = y_source3.unsqueeze(0).float().to(self.device)
                if self.patch_size:
                    x_source1 = x_source1.view(-1,1,self.patch_size,self.patch_size)
                    y_source1 = y_source1.view(-1,1,self.patch_size,self.patch_size)
                    x_source2 = x_source2.view(-1,1,self.patch_size,self.patch_size)
                    y_source2 = y_source2.view(-1,1,self.patch_size,self.patch_size)
                    x_source3 = x_source3.view(-1,1,self.patch_size,self.patch_size)
                    y_source3 = y_source3.view(-1,1,self.patch_size,self.patch_size)

                #generate
                pred_source1,pred_source2,pred_source3 = self.model(x_source1, x_source2,x_source3)
                ssim_source1 = 1-ssim(pred_source1,y_source1)

                loss_source1 = self.criterion(pred_source1,y_source1)
                ssim_source2 = 1-ssim(pred_source2,y_source2)

                loss_source2 = self.criterion(pred_source2,y_source2)
                ssim_source3 = 1-ssim(pred_source3,y_source3)

                loss_source3 = self.criterion(pred_source3,y_source3)
                ssim_loss =0.001* (ssim_source1 + ssim_source2 + ssim_source3)

                l1_loss = loss_source1 + loss_source2 + loss_source3
                total_loss = l1_loss + ssim_loss #+ lipis_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()    # learning decreasing
                # 打印，保存模型
                if iter % self.print_iters == 0 and iter !=0:
                    print('total_iter: [{}],  epoch: [{}/{}],  iter: [{}/{}], lr: [{:.30f}]， total_loss: [{:.8f}] \n'
                              'AAPM_loss: {:.8f},timo_loss: {:.8f}, head_loss: {:.8f} \n l1_loss: {:.8f} ,  ssim_losss: {:.8f},  time:{:.1f}s'.format(total_iter,epoch, self.num_epochs, iter,
                              len(self.data_loader), self.optimizer.state_dict()['param_groups'][0]['lr'],total_loss.item(),
                               loss_source1.item(),loss_source2.item(),loss_source3.item(),l1_loss.item(),ssim_loss.item(),time.time()-start_time))
                if total_iter % self.save_iters ==0:
                    self.save_model(total_iter)
            #验证
            if total_iter % self.save_iters==0 and total_iter!=0:
                dataset_ =ct_dataset(mode='test',load_mode=0,augment=False,saved_path='./val',test_patient='val',
                                     patch_n=None,patch_size=None,transform=False)
                data_loade = DataLoader(dataset=dataset_,batch_size=1,shuffle=False,num_workers=8)
                self.model_val = MSDnet().to(self.device)
                f = os.path.join(self.save_path,'model_{}iter.ckpt'.format((epoch+1)*self.save_iters))
                if self.multi_gpu:
                    state_d = OrderedDict()
                    for k,v in torch.load(f):
                        n = k[7:]
                        state_d[n]=v
                    self.model_val.load_state_dict(state_d)
                else:
                    self.model_val.load_state_dict(torch.load(f))
                with torch.no_grad():
                    for i, ((x_source1, y_source1), (x_source2, y_source2), (x_source3, y_source3)) in enumerate(data_loade):
                        shape_ = x_source1.shape[-1]
                        x_source1 = x_source1.unsqueeze(0).float().to(self.device)
                        y_source1 = y_source1.unsqueeze(0).float().to(self.device)
                        x_source2 = x_source2.unsqueeze(0).float().to(self.device)
                        y_source2 = y_source2.unsqueeze(0).float().to(self.device)
                        x_source3 = x_source3.unsqueeze(0).float().to(self.device)
                        y_source3 = y_source3.unsqueeze(0).float().to(self.device)
                        pred_source1,pred_source2,pred_source3 = self.model(x_source1, x_source2,x_source3)
                        data_range = self.trunc_max - self.trunc_min
                        x1_source1 = self.trunc(self.denormalize_(x_source1))
                        y1_source1 = self.trunc((self.denormalize_(y_source1)))
                        pred1_source1 = self.trunc(self.denormalize_(pred_source1))
                        original_result_source1, pred_result_source1 = compute_measure(x1_source1, y1_source1, pred1_source1, data_range)
                        pred_psnr_avg_source1 += pred_result_source1[0]
                        pred_ssim_avg_source1 += pred_result_source1[1]
                        pred_rmse_avg_source1 += pred_result_source1[2]
                        x1_source2 = self.trunc(self.denormalize_(x_source2))
                        y1_source2 = self.trunc((self.denormalize_(y_source2)))
                        pred1_source2 = self.trunc(self.denormalize_(pred_source2))
                        original_result_source2, pred_result_source2 = compute_measure(x1_source2, y1_source2, pred1_source2, data_range)
                        pred_psnr_avg_source2 += pred_result_source2[0]
                        pred_ssim_avg_source2 += pred_result_source2[1]
                        pred_rmse_avg_source2 += pred_result_source2[2]
                        x1_source3 = self.trunc(self.denormalize_(x_source3))
                        y1_source3 = self.trunc((self.denormalize_(y_source3)))
                        pred1_source3 = self.trunc(self.denormalize_(pred_source3))
                        original_result_source3, pred_result_source3 = compute_measure(x1_source3, y1_source3, pred1_source3, data_range)
                        pred_psnr_avg_source3 += pred_result_source3[0]
                        pred_ssim_avg_source3 += pred_result_source3[1]
                        pred_rmse_avg_source3 += pred_result_source3[2]

                with open('./save/pred_psnr_avg.txt', 'a') as f:
                    f.write('EPOCH:%d total:%.20f    AAPM:%.20f    timo:%.20f    head:%.20f' % (epoch,
                                                                                                    (pred_psnr_avg_source1+pred_psnr_avg_source2+pred_psnr_avg_source3) / (3*len(data_loade)),
                                                                                                    pred_psnr_avg_source1/len(data_loade),pred_psnr_avg_source2/len(data_loade),
                                                                                                    pred_psnr_avg_source3/len(data_loade)) + '\n')
                    f.close()
                with open('./save_80/pred_ssim_avg.txt', 'a') as f:
                    f.write('EPOCH:%d total:%.20f    AAPM:%.20f    timo:%.20f     head:%.20f' % (epoch,
                                                                                                    (pred_ssim_avg_source1+pred_ssim_avg_source2+pred_ssim_avg_source3) / (3*len(data_loade)),
                                                                                                    pred_ssim_avg_source1/len(data_loade),pred_ssim_avg_source2/len(data_loade),
                                                                                                    pred_ssim_avg_source3/len(data_loade)) + '\n')
                    f.close()
                with open('./save_80/pred_rmse_avg.txt', 'a') as f:
                    f.write('EPOCH:%d total:%.20f    AAPM:%.20f    timo:%.20f     head:%.20f' % (epoch,
                                                                                                    (pred_rmse_avg_source1+pred_rmse_avg_source2+pred_rmse_avg_source3) / (3*len(data_loade)),
                                                                                                    pred_rmse_avg_source1/len(data_loade),pred_rmse_avg_source2/len(data_loade),
                                                                                                    pred_rmse_avg_source3/len(data_loade)) + '\n')
                    f.close()
                pred_psnr_avg_source1, pred_ssim_avg_source1, pred_rmse_avg_source1 = 0, 0, 0
                pred_psnr_avg_source2, pred_ssim_avg_source2, pred_rmse_avg_source2 = 0, 0, 0
                pred_psnr_avg_source3, pred_ssim_avg_source3, pred_rmse_avg_source3 = 0, 0, 0
            else:
                continue

    def test(self):
        del self.model
        # load
        self.model =MSDnet().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg_source1, ori_ssim_avg_source1, ori_rmse_avg_source1 = 0, 0, 0
        ori_psnr_avg_source2, ori_ssim_avg_source2, ori_rmse_avg_source2 = 0, 0, 0
        ori_psnr_avg_source3, ori_ssim_avg_source3, ori_rmse_avg_source3 = 0, 0, 0
        pred_psnr_avg_source1, pred_ssim_avg_source1, pred_rmse_avg_source1 = 0, 0, 0
        pred_psnr_avg_source2, pred_ssim_avg_source2, pred_rmse_avg_source2 = 0, 0, 0
        pred_psnr_avg_source3, pred_ssim_avg_source3, pred_rmse_avg_source3 = 0, 0, 0
        # print('self.data_loader=', list(self.data_loader)[0])
        # for i, (x, y) in enumerate(self.data_loader):
        # print('i=',i)
        # print('(x, y)=
        ori_psnr_source1, ori_ssim_source1, ori_rmse_source1 = [],[],[]
        ori_psnr_source2, ori_ssim_source2, ori_rmse_source2 = [],[],[]
        ori_psnr_source3, ori_ssim_source3, ori_rmse_source3 = [],[],[]
        pred_psnr_source1, pred_ssim_source1, pred_rmse_source1 = [],[],[]
        pred_psnr_source2, pred_ssim_source2, pred_rmse_source2 = [],[],[]
        pred_psnr_source3, pred_ssim_source3, pred_rmse_source3 = [],[],[]

        with torch.no_grad():
            for i, ((x_source1, y_source1), (x_source2, y_source2), (x_source3, y_source3)) in enumerate(self.data_loader):
                shape_ = x_source1.shape[-1]
                x_source1 = x_source1.unsqueeze(0).float().to(self.device)
                y_source1 = y_source1.unsqueeze(0).float().to(self.device)
                x_source2 = x_source2.unsqueeze(0).float().to(self.device)
                y_source2 = y_source2.unsqueeze(0).float().to(self.device)
                x_source3 = x_source3.unsqueeze(0).float().to(self.device)
                y_source3 = y_source3.unsqueeze(0).float().to(self.device)
                pred_source1,pred_source2,pred_source3 = self.model(x_source1, x_source2,x_source3)


                x_source1 = self.trunc(self.denormalize_(x_source1.view(shape_, shape_).cpu().detach()))
                y_source1 = self.trunc(self.denormalize_(y_source1.view(shape_, shape_).cpu().detach()))
                pred_source1 = self.trunc(self.denormalize_(pred_source1.view(shape_, shape_).cpu().detach()))
                x_source2 = self.trunc(self.denormalize_(x_source2.view(shape_, shape_).cpu().detach()))
                y_source2 = self.trunc(self.denormalize_(y_source2.view(shape_, shape_).cpu().detach()))
                pred_source2 = self.trunc(self.denormalize_(pred_source2.view(shape_, shape_).cpu().detach()))
                x_source3 = self.trunc(self.denormalize_(x_source3.view(shape_, shape_).cpu().detach()))
                y_source3 = self.trunc(self.denormalize_(y_source3.view(shape_, shape_).cpu().detach()))
                pred_source3 = self.trunc(self.denormalize_(pred_source3.view(shape_, shape_).cpu().detach()))

                np.save(os.path.join(self.save_path, 'x_source1', '{}_result'.format(i)), x_source1)
                np.save(os.path.join(self.save_path, 'y_source1', '{}_result'.format(i)), y_source1)
                np.save(os.path.join(self.save_path, 'pred_source1', '{}_result'.format(i)), pred_source1)
                np.save(os.path.join(self.save_path, 'x_source2', '{}_result'.format(i)), x_source2)
                np.save(os.path.join(self.save_path, 'y_source2', '{}_result'.format(i)), y_source2)
                np.save(os.path.join(self.save_path, 'pred_source2', '{}_result'.format(i)), pred_source2)
                np.save(os.path.join(self.save_path, 'x_source3', '{}_result'.format(i)), x_source3)
                np.save(os.path.join(self.save_path, 'y_source3', '{}_result'.format(i)), y_source3)
                np.save(os.path.join(self.save_path, 'pred_source3', '{}_result'.format(i)), pred_source3)

                # # 存放迭代,金标准
                # np.save(os.path.join(self.save_path, 'iter', 'LDCT_{}_input'.format(i)), pred1)
                # np.save(os.path.join(self.save_path, 'iter', 'LDCT_{}_target'.format(i)), y1)

                # 存放迭代,无金标准
                # np.save(os.path.join(self.save_path, 'iter', 'LDCT_{}_input'.format(i)), pred1)
                # np.save(os.path.join(self.save_path, 'iter', 'LDCT_{}_target'.format(i)), pred1)
                data_range = self.trunc_max - self.trunc_min
                original_result_source1, pred_result_source1 = compute_measure(x_source1, y_source1, pred_source1, data_range)
                ori_psnr_avg_source1 += original_result_source1[0]
                ori_ssim_avg_source1 += original_result_source1[1]
                ori_rmse_avg_source1 += original_result_source1[2]
                pred_psnr_avg_source1 += pred_result_source1[0]
                pred_ssim_avg_source1 += pred_result_source1[1]
                pred_rmse_avg_source1 += pred_result_source1[2]

                ori_psnr_source1.append(original_result_source1[0])
                ori_ssim_source1.append(original_result_source1[1])
                ori_rmse_source1.append(original_result_source1[2])
                pred_psnr_source1.append(pred_result_source1[0])
                pred_ssim_source1.append(pred_result_source1[1])
                pred_rmse_source1.append(pred_result_source1[2])

                original_result_source2, pred_result_source2 = compute_measure(x_source2, y_source2, pred_source2, data_range)
                pred_psnr_avg_source2 += pred_result_source2[0]
                pred_ssim_avg_source2 += pred_result_source2[1]
                pred_rmse_avg_source2 += pred_result_source2[2]
                ori_psnr_avg_source2 += original_result_source2[0]
                ori_ssim_avg_source2+= original_result_source2[1]
                ori_rmse_avg_source2 += original_result_source2[2]


                ori_psnr_source2.append(original_result_source2[0])
                ori_ssim_source2.append(original_result_source2[1])
                ori_rmse_source2.append(original_result_source2[2])
                pred_psnr_source2.append(pred_result_source2[0])
                pred_ssim_source2.append(pred_result_source2[1])
                pred_rmse_source2.append(pred_result_source2[2])

                original_result_source3, pred_result_source3 = compute_measure(x_source3, y_source3, pred_source3, data_range)
                pred_psnr_avg_source3 += pred_result_source3[0]
                pred_ssim_avg_source3 += pred_result_source3[1]
                pred_rmse_avg_source3 += pred_result_source3[2]
                ori_psnr_avg_source3 += original_result_source3[0]
                ori_ssim_avg_source3 += original_result_source3[1]
                ori_rmse_avg_source3 += original_result_source3[2]


                ori_psnr_source3.append(original_result_source3[0])
                ori_ssim_source3.append(original_result_source3[1])
                ori_rmse_source3.append(original_result_source3[2])
                pred_psnr_source3.append(pred_result_source3[0])
                pred_ssim_source3.append(pred_result_source3[1])
                pred_rmse_source3.append(pred_result_source3[2])

                # save result figure
                if self.result_fig:
                    self.save_fig(x_source1, y_source1, pred_source1, i, original_result_source1, pred_result_source1,'fig_source1')
                    self.save_fig(x_source2, y_source2, pred_source2, i, original_result_source2, pred_result_source2,'fig_source2')
                    self.save_fig(x_source3, y_source3, pred_source3, i, original_result_source3, pred_result_source3,'fig_source3')

            print('\n')
            print('Original\nPSNR avg: {:.4f}, PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,SSIM avg: {:.4f} \nRMSE avg: {:.4f},SSIM avg: {:.4f} '.format(
                ori_psnr_avg_source1 / len(self.data_loader),np.std(ori_psnr_source1),
                ori_ssim_avg_source1 / len(self.data_loader),np.std(ori_ssim_source1),
                ori_rmse_avg_source1 / len(self.data_loader),np.std(ori_rmse_source1)))
            print('After learning\nPSNR avg: {:.4f},PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,PSNR avg: {:.4f}\nRMSE avg: {:.4f},PSNR avg: {:.4f}'.format(
                pred_psnr_avg_source1 / len(self.data_loader),np.std(pred_psnr_source1),
                pred_ssim_avg_source1 / len(self.data_loader),np.std(pred_ssim_source1),
                pred_rmse_avg_source1 / len(self.data_loader),np.std(pred_rmse_source1)))
            print('\n')
            print('Original\nPSNR avg: {:.4f}, PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,SSIM avg: {:.4f} \nRMSE avg: {:.4f},SSIM avg: {:.4f}'.format(
                ori_psnr_avg_source2 / len(self.data_loader), np.std(ori_psnr_source2),ori_ssim_avg_source2 / len(self.data_loader),np.std(ori_ssim_source2),
                ori_rmse_avg_source2 / len(self.data_loader),np.std(ori_rmse_source2)))
            print('After learning\nPSNR avg: {:.4f},PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,PSNR avg: {:.4f}\nRMSE avg: {:.4f},PSNR avg: {:.4f}'.format(
                pred_psnr_avg_source2 / len(self.data_loader), np.std(pred_psnr_source2),pred_ssim_avg_source2 / len(self.data_loader),np.std(pred_ssim_source2),
                pred_rmse_avg_source2 / len(self.data_loader),np.std(pred_rmse_source2)))
            print('\n')
            print('Original\nPSNR avg: {:.4f}, PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,SSIM avg: {:.4f} \nRMSE avg: {:.4f},SSIM avg: {:.4f}'.format(
                ori_psnr_avg_source3 / len(self.data_loader),np.std(ori_psnr_source3),
                ori_ssim_avg_source3 / len(self.data_loader),np.std(ori_ssim_source3),
                ori_rmse_avg_source3/ len(self.data_loader),np.std(ori_rmse_source3)))
            print('After learning\nPSNR avg: {:.4f},PSNR avg: {:.4f} \nSSIM avg: {:.4f} ,PSNR avg: {:.4f}\nRMSE avg: {:.4f},PSNR avg: {:.4f}'.format(
                pred_psnr_avg_source3 / len(self.data_loader), np.std(pred_psnr_source3),
                pred_ssim_avg_source3 / len(self.data_loader),np.std(pred_ssim_source3),
                pred_rmse_avg_source3 / len(self.data_loader),np.std(pred_rmse_source3)))


