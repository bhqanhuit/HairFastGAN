import argparse
import os
import sys
from pathlib import Path
import torch

from torchvision.utils import save_image
from tqdm.auto import tqdm
from scale_images import image_scale

from hair_swap import HairFast, get_parser
import time
import cv2
import torchvision.transforms as T
from PIL import Image

def ImageFolderUtils(dir_path):
    out_path = 'datasets/FFHQ_Resized'
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        img = img[:, 0:1024, :]
        cv2.imwrite(out_path + '/' + img_path, img)
        # print(img.shape)
        # exit()
    print('done')


def main(model_args, args):
    start_time = time.time()
    hair_fast = HairFast(model_args)
    origin_scale = 'datasets/FFHQ_TrueScale'
    # dir_path = 'datasets/FFHQ_Resized'
    # ImageFolderUtils(dir_path)
    
    
    with open('datasets/testPair.txt') as file:
        lines = [line.rstrip() for line in file]
        cnt = 0
        for line in lines:
            cnt += 1
            source, shape = line.split(' ')
            color = shape

            face_path = os.path.join(origin_scale, source)
            shape_path = os.path.join(origin_scale, shape)
            color_path = os.path.join(origin_scale, color)
            print(face_path, shape_path)
            scaled_img = image_scale(shape_path, face_path).to('cuda')
            
            transform = T.Compose([T.ToTensor()])
            source_im = transform(Image.open(face_path)).to('cuda')
            shape_im = transform(Image.open(shape_path)).to('cuda')
            color_im = transform(Image.open(shape_path)).to('cuda')

            transform_toPIL = T.ToPILImage()

            final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=None)
            # final_image = hair_fast.swap(transform_toPIL(scaled_img), transform_toPIL(shape_im), transform_toPIL(color_im), benchmark=args.benchmark, exp_name=None)
            if (cnt == 31): exit()
            # save_image(final_image, 'test_outputs_FaceScale/' + str(cnt).zfill(10) + '.jpg')
            # save_image(torch.cat([final_image, source_im, shape_im, scaled_img], dim=2), 'test_outputsFull_FaceScale/' + str(cnt).zfill(10) + '.jpg')
            # exit()
            # break

    

    # experiments: list[str | tuple[str, str, str]] = []
    # if args.file_path is not None:
    #     with open(args.file_path, 'r') as file:
    #         experiments.extend(file.readlines())

    # if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
    #     experiments.append((args.face_path, args.shape_path, args.color_path))

    # print('start')
    # for exp in tqdm(experiments):
    #     if isinstance(exp, str):
    #         file_1, file_2, file_3 = exp.split()
    #     else:
    #         file_1, file_2, file_3 = exp

    #     face_path = args.input_dir / file_1
    #     shape_path = args.input_dir / file_2
    #     color_path = args.input_dir / file_3

    #     base_name = '_'.join([path.stem for path in (face_path, shape_path, color_path)])
    #     exp_name = base_name if model_args.save_all else None

    #     if isinstance(exp, str) or args.result_path is None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    #         output_image_path = args.output_dir / f'{base_name}.png'
    #     else:
    #         os.makedirs(args.result_path.parent, exist_ok=True)
    #         output_image_path = args.result_path

    #     final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=exp_name)
    #     save_image(final_image, output_image_path)
    #     print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--input_dir', type=Path, default='', help='The directory of the images to be inverted')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for a set of experiments
    parser.add_argument('--file_path', type=Path, default=None,
                        help='File with experiments with the format "face_path.png shape_path.png color_path.png"')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default=None, help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default=None, help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default=None, help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default=None, help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)
