from aBack.core.valer import Valer

if __name__ == '__main__':
    # logdir = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_2022-06-09-12-49_576x576'
    logdir = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_v3fullconv_2022-07-13-22-04-576x576_12m'
    # logdir = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_v3_2022-07-01-23-30-576x576_12m'
    # logdir = r'E:\work\Master_Seg\Space_seg\result\Seg_model_baseline_v2_2022-06-21-14-47_576x576'

    testset_dir = r'E:\Dataset\SegData\Testset12M'
    valer = Valer(logdir, test_dir=testset_dir, front_dist=12, img_size=(576, 576))

    valer.eval_acc_576x576_12m(dist_range=(0, 4), contain_freespace=False)

    # valer.inference_frame(frame_dir="E:\Dataset\SegData\Testset\[test1]-[n1541]\image")
