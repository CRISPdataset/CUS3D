# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import sys
sys.path.append(os.getcwd())
import time
import cv2
import tqdm
from PIL import Image
from detectron2.config import get_cfg
import numpy as np
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo, SAMVisualizationDemo

# constants
WINDOW_NAME = "Open vocabulary segmentation"

# python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'chair' 'table' 'curtain' 'Flooring' 'paper'  --input ./resources/demo_samples/0_0.jpg --output ./pred --opts MODEL.WEIGHTS pth/ovseg_swinbase_vitL14_ft_mpt.pth

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--class-lebels",
        help="class label file path",
        metavar="FILE",
        default='',
    )
    return parser

import pandas as pd
import sys

def prompt_func(class_names, predictor):
    templates = ["itap of a {}.",
                "a bad photo of the {}.",
                "a origami {}.",
                "a photo of the large {}.",
                "a {} in a video game.",
                "art of the {}.",
                "a photo of the small {}."
                ]
    zeroshot_weights = []
    for classname in tqdm(class_names):
        texts = [template.format(classname) for template in templates] #format with class
        text_features = predictor.get_text_features(texts)
        print(text_features.shape)
        # text_features = text_features[:-1, :]
        
from data.scannetv2.conf import CLASS_LABELS_20

def get_cos(image_features, text_features):
    image_features = image_features.permute(1, 2, 0)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # [240, 320, 4]
    values, indices = similarity[:, :].topk(1)
    return values, indices
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    class_names = args.class_names
    # args.class_labels = '/home/wxj/code/P2P/ov-seg/resources/scannetv2-labels.combined.tsv'
    args.class_labels = None
    if args.class_labels:
        tsv_file = pd.read_csv(
            args.class_labels,
            sep='\t',
            header=0,
            index_col='id'
        ) 
        class_names = tsv_file['category'].tolist()
    class_names = list(CLASS_LABELS_20)    
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, image_features, text_features, odp_fc = demo.run_on_image(img, class_names)
            text_features = text_features[:-1]
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                
                # print(fc.shape)
                # print(text_fc.shape)

                prob, indices = get_cos(odp_fc, text_features)
                
                img = demo.vis(img, class_names, indices.squeeze().cpu())
                
                
                # indices = indices * 255 / (len(class_names)-1)
                # ind = indices.squeeze().cpu().numpy().astype(np.uint8)
                # img = Image.fromarray(ind, mode='L')
                out_filename = os.path.join(args.output, 'fc_'+os.path.basename(path))
                img.save(out_filename)
                
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        raise NotImplementedError