import os, argparse
from PIL import Image
import matplotlib.pyplot as plt

from utils.evaluator import evalutate
from utils.utils import overlay




def get_args():
    parser = argparse.ArgumentParser(description='Dataloader test')

    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--ngpu', default=1, type=int, help='gpu num')
    parser.add_argument('--workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--clip_model', default='ViT-B/16', type=str, help='clip model RN50 RN101 ViT-B/32')
    parser.add_argument('--nb_epoch', default=32, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.000025, type=float, help='batch size 16 learning rate')
    parser.add_argument('--power', default=0.1, type=float, help='lr poly power')
    parser.add_argument('--steps', default=[15, 28], type=int, nargs='+', help='in which step lr decay by power')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--size', default=416, type=int, help='image size')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog/grefcoco')

    parser.add_argument('--num_query', default=16, type=int, help='the number of query')
    parser.add_argument('--w_seg', default=0.1, type=float, help='weight of the seg loss')
    parser.add_argument('--w_coord', default=5, type=float, help='weight of the reg loss')
    parser.add_argument('--tunelang', dest='tunelang', default=True, action='store_true', help='if finetune language model')
    parser.add_argument('--anchor_imsize', default=416, type=int, help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data', help='location of pre-parsed dataset info')
    parser.add_argument('--time', default=15, type=int, help='maximum time steps (lang length) per batch')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to log directory')

    parser.add_argument('--fusion_dim', default=768, type=int, help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='/home/cid2r/GitHub/ASDA/saved_models/savename_model_best.pth.tar', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')

    parser.add_argument('--seg_thresh', default=0.35, type=float, help='seg score above this value means foreground')
    parser.add_argument('--seg_out_stride', default=2, type=int, help='the seg out stride')
    
    #####
    parser.add_argument('--img_path', default='input.jpg', type=str, help='Set Image file')
    parser.add_argument('--wav_path', default='input.wav', type=str, help='Set Wave file')
    parser.add_argument('--resize_mode', default='letterbox', type=str, choices=['letterbox', 'center_crop'],
                        help='Choose resize method: letterbox or center_crop_resize')
    parser.add_argument('--method', default='cross_modal', type=str, choices=['cross_modal', 'caption', 'classify'],
                        help='Choose evaluation method: cross_modal, caption, classify')
    #####

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    args.lr = args.lr * (args.batch_size * args.ngpu // 16)

    return args



if __name__ == '__main__':
    args = get_args()
    
    pred = evalutate(args)

    image = Image.open(args.img_path).convert("RGB")
    
    plt.imshow(overlay(image, pred.squeeze().cpu().detach().numpy()))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    save_path = args.img_path[:-4] + '_output.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print("Output file saved at: " + save_path)

