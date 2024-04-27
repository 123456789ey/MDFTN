import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main(args):
    cudnn.benchmark = True   #

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fname = 'fig_source1','fig_source2','fig_source3'
        xname = 'x_source1','x_source2','x_source3'
        yname = 'y_source1','y_source2','y_source3'
        pname = 'pred_source1','pred_source2','pred_source3'
        for i in range(3):
            fig_path = os.path.join(args.save_path, fname[i])
            x_path = os.path.join(args.save_path, xname[i])
            y_path = os.path.join(args.save_path, yname[i])
            pred_path = os.path.join(args.save_path, pname[i])  #pred预测结果
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
                print('Create path : {}'.format(fig_path))
            if not os.path.exists(x_path):
                os.makedirs(x_path)
                print('Create path : {}'.format(x_path))
            if not os.path.exists(y_path):
                os.makedirs(y_path)
                print('Create path : {}'.format(y_path))
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
                print('Create path : {}'.format(pred_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             augment= False,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()     #创建一个解析对象
    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0)   #load_mode
    parser.add_argument('--augment', type=bool, default=True)
    #parser.add_argument('--saved_path', type=str, default='./train') #7645
    parser.add_argument('--saved_path', type=str, default='./test')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--test_patient', type=str, default='test')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--norm_range_min', type=float, default=-800.0)
    parser.add_argument('--norm_range_max', type=float, default=1000.0)
    parser.add_argument('--trunc_min', type=float, default=-800.0)
    parser.add_argument('--trunc_max', type=float, default=1000.0)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--patch_n', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=80)
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=800*20)
    parser.add_argument('--save_iters', type=int, default=800)
    parser.add_argument('--test_iters', type=int, default=800*91)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_d_train', type=int, default=4)
    parser.add_argument('--lambda_', type=float, default=10.0)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()    #进行解析
    print(args)
    main(args)
