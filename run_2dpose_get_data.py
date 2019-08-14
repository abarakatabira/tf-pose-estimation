import argparse
import cv2
import logging
import pandas as pd
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--movie', type=str, default='./movie/test.mp4')
    parser.add_argument('--dataname',type=str,default='')
    parser.add_argument('--datas', type=str, default='./movie/data/')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--frame', type=float, default=1,
                        help='the frame parcentage of total frame count')
    parser.add_argument('--frameterm', type=int, default=1,
                        help='frame term')

    args = parser.parse_args()
    movie = cv2.VideoCapture(args.movie)

    # get total frame count
    count = movie.get(cv2.CAP_PROP_FRAME_COUNT)

    w, h = model_wh(args.resize)

    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # get 2d estimation result
    dfs = pd.DataFrame(index=[])
    columns = ['flame', 'human', 'point', 'x', 'y']

    # per frame
    for i in range(0,int(args.frame*count)):
        _, frame = movie.read()

        #only get per frameterm
        if i % int(args.frameterm) != 0:
            continue

        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
        image_h, image_w = frame.shape[:2]

        # per human
        xx = 0
        for human in humans:
            xx = xx + 1

            # per body_part
            for m in human.body_parts:
                body_part = human.body_parts[m]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                list = [[i, xx, m, center[0],center[1]]]
                df = pd.DataFrame(data=list, columns=columns)

                # add list to dfs
                dfs = pd.concat([dfs, df])

        #get image if needed
        cv2.imwrite(args.datas + args.dataname +'_'+ args.movie[-len(args.movie)+8:-4] + '_'  + str(i) +  '_data.jpg', frame)

    # result output
    dfs.to_csv(args.datas + args.dataname +'_'+ args.movie[-len(args.movie)+8:-4]  +  '_2ddata.csv', encoding="utf-8")