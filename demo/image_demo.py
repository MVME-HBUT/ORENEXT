from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config = '/home/yzhang/AS-MLP-Object-Detection/configs/asmlp_pyt/orenext/orenext_stonemlp_sparsefc_ptsdml.py'
checkpoint = '/home/yzhang/AS-MLP-Object-Detection/work_dirs/orenext_stonemlp_sparsefc_ptsdml1.py/epoch_11.pth'
# config = '/home/yzhang/AS-MLP-Object-Detection/configs/asmlp_pyt/point/point_rend_asmlp1.py'
# checkpoint = '/home/yzhang/AS-MLP-Object-Detection/work_dirs/point/point_rend_asmlp/epoch_3.pth'

img = './test_demo/043.png'
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device='cuda:0')
    # test a single image
    result = inference_detector(model, img)
    # show the results
    model.show_result(
        img,
        result,
        score_thr=0.6,
        show=True,
        out_file='result2.jpg',
        
        text_color='green',
        thickness=2,  
        font_size=10)



if __name__ == '__main__':
    main()


#  python demo/image_demo.py test_demo/2.png configs/asmlp_pyt/orenext_stonemlp_sparsefc_ptsdml.py work_dirs/orenext_stonemlp_sparsefc_ptsdml.py/epoch_6.pth --device cpu