from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    try:
        eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)

def demo(opt):
    result_root = opt.output_root
    mkdir_if_missing(result_root)
    print('Starting tracking...')
    list_dir = os.listdir(opt.input_image)
    list_dir.sort(key=lambda x: int(x[5:]))
    dataloader = datasets.LoadImages(opt.input_image, opt.img_size)  # 此处已经获取所有测试文件夹和图片
    for subdir in list_dir:
        result_filename = os.path.join(result_root, '%s.txt' % subdir)
        # frame_rate = dataloader.frame_rate
        frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame_%s'%subdir)
        eval_seq(opt, dataloader, 'mot', result_filename,
                save_dir=frame_dir, show_image=False)
        dataloader.index += 1
        BaseTrack._count = 0

        #  try/except语句用来检测try语句块中的错误，从而让except语句捕获异常信息并处理。

        if opt.output_format == 'video':
            output_video_path = os.path.join(result_root, 'result_%s.mp4'%subdir)
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b:v 5000k -c:v mpeg4 {}'\
                .format(os.path.join(result_root, 'frame_%s'%subdir), output_video_path)
            os.system(cmd_str)

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
