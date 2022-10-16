from aBack.core.valer import Valer

if __name__ == '__main__':
    # logdir = r'E:\work\pytorch_training_tool_seg\Space_seg\result\from_remote\Seg_hrnet32_2022-04-30-17-04'
    # logdir = r'E:\work\pytorch_training_tool_seg\Space_seg\run\Seg_hrnet48_2022-04-18-00-34'
    # logdir = r'E:\work\pytorch_training_tool_seg\Space_seg\run\Seg_STDCNetLarge_2022-04-21-17-59'
    # logdir = r'E:\work\pytorch_training_tool_seg\Space_seg\result\Seg_model_baseline_2022-05-19-12-39'
    logdir = r'E:\work\Master_Seg\Space_seg\run\Seg_model_baseline_v1_2022-06-13-19-23'

    testset_dir = r'E:\Dataset\SegData\Testset'
    valer = Valer(logdir, test_dir=testset_dir, img_size=(576, 576))

    valer.eval_acc(contain_freespace=False)

    # valer.inference_frame(frame_dir="E:\Dataset\SegData\Testset\[test1]-[n1541]\image")
