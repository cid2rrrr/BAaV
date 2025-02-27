import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import utils.utils as utils
from utils.evaluator import evalutate



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')
    # Dataset
    parser.add_argument('--testset', default='is3', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='/data_2T/datasets/', type=str, help='Root directory path of data')
    parser.add_argument('--visualize', action='store_true')
    
    ####
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
    parser.add_argument('--resize_mode', default='center_crop', type=str, choices=['letterbox', 'center_crop'],
                        help='Choose resize method: letterbox or center_crop_resize')
    parser.add_argument('--method', default='cross_modal', type=str, choices=['cross_modal', 'caption', 'classify'],
                        help='Choose evaluation method: cross_modal, caption, classify')
    
    return parser.parse_args()


@torch.no_grad()
def main(args):
    evaluator = utils.Evaluator_iiou()

    img_dir = os.path.join(args.test_data_path, "img")
    wav_dir = os.path.join(args.test_data_path, "wav")
    gt_dir = os.path.join(args.test_data_path, "gt_seg")

    img_ext = [".jpg", ".png"]
    wav_ext = ".wav"
    gt_ext = [".jpg", ".png"]

    file_stems = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}

    datasets = [
        (
            next(os.path.join(img_dir, stem + ext) for ext in img_ext if os.path.exists(os.path.join(img_dir, stem + ext))),
            os.path.join(wav_dir, stem + wav_ext),
            next(os.path.join(gt_dir, stem + ext) for ext in gt_ext if os.path.exists(os.path.join(gt_dir, stem + ext)))
        )
        for stem in sorted(file_stems)
    ]

    # datasets = [('./input.jpg', './input.wav', 'input_gt.jpg')]
    for img_path, wav_path, gt_path in tqdm(datasets, desc="Processing files"):

        pred = evalutate(args, img_path, wav_path)
        pred = pred.squeeze().cpu().detach().numpy()

        gt_map = np.array(Image.open(gt_path).resize((224, 224)))
        gt_map[gt_map < 128] = 0
        gt_map[gt_map >= 128] = 1

        threshold = 0.5
        evaluator.cal_CIOU(pred, gt_map, threshold)

        gt_nums = (gt_map != 0).sum()
        if int(gt_nums) == 0:
            gt_nums = int(pred.shape[0] * pred.shape[1]) // 2
        threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1]) - int(gt_nums)]
        evaluator.cal_CIOU_adap(pred, gt_map, threshold)

        if args.visualize:
            target_name = img_path.split('/')[-1]
            frame_ori = Image.open(img_path).resize((224, 224))
            plt.imshow(utils.overlay(frame_ori, pred).resize((224, 224)))
            plt.axis('off')
            plt.savefig(os.path.join('../unified_qualitatives', args.testset, target_name), bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

    print('CIoU_adap : {:.1f}'.format(evaluator.finalize_cIoU()[1] * 100))
    print('AUC_adap : {:.1f}'.format(evaluator.finalize_AUC()[1] * 100))
    print('mIoU_adap : {:.3f}'.format(np.mean(evaluator.ciou)))
    print('F-Score_adap : {:.3f}'.format(np.mean(evaluator.fscore_adap)))
    print('Jaccard : {:.3f}'.format(np.mean(evaluator.ciou_adap)))
    print('F-score : {:.3f}'.format(np.mean(evaluator.fscore)))


if __name__ == '__main__':
    args = get_arguments()
    main(args)
