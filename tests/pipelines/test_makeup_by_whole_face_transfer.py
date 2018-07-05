import unittest
import cv2
import os
import numpy as np

from pipelines import makeup_by_whole_face_transfer


class MyTestCase(unittest.TestCase):
    def test_makeup_by_whole_face_transfer(self):
        example_name = 'example_0'
        example_face_path = os.path.join('../../assets/example_face', example_name + '.jpg')
        example_face = cv2.imread(example_face_path)

        example_alpha_path = os.path.join('../../assets/example_alpha', example_name + '.jpg')
        example_face_alpha = cv2.imread(example_alpha_path)

        example_landmarks_path = os.path.join('../../assets/example_landmarks', example_name + '.txt')
        example_landmarks = np.loadtxt(example_landmarks_path, dtype=int)
        target_path = '../assets/targets/target_1.jpg'
        target = cv2.imread(target_path)

        result = makeup_by_whole_face_transfer.makeup_by_whole_face_transfer(target,
                                                                             example_face,
                                                                             example_face_alpha,
                                                                             example_landmarks,
                                                                             rectify_landmarks=False)
        cv2.imwrite('test_whole_face_transfer.jpg', result)

    def test_pre_proc_example(self):
        example_folder_path = '../assets/examples/'
        examples_path = os.listdir(example_folder_path)
        examples = [example for example in examples_path if example.endswith('.jpg') | example.endswith('.jpeg')]
        root = '../../assets'

        for example in examples:
            example_res_folder_path = os.path.join(root, example.split('.')[0])
            if os.path.exists(example_res_folder_path) is False:
                os.mkdir(example_res_folder_path)
            example_image = cv2.imread(os.path.join(example_folder_path, example))
            face, alpha, landmarks = makeup_by_whole_face_transfer.pre_proc_example(example_image)
            cv2.imwrite(os.path.join(example_res_folder_path, 'face_' + example.split('.')[0]+'.jpg'), np.array(face, dtype=np.uint8))
            cv2.imwrite(os.path.join(example_res_folder_path, 'face_alpha_' + example.split('.')[0]+'.jpg'), np.array(255*alpha, dtype=np.uint8))
            np.savetxt(os.path.join(example_res_folder_path, example.split('.')[0]+'.txt'), landmarks, fmt='%i')

    def test_makeup(self):
        example_folder_path = '../assets/examples'
        examples_path = os.listdir(example_folder_path)
        examples = [example for example in examples_path if example.endswith('.jpg') | example.endswith('.jpeg')]

        target_folder_path = '../assets/targets'
        targets_path = os.listdir(target_folder_path)
        targets = [target for target in targets_path if target.endswith('.jpg') | target.endswith('.jpeg')]

        output_folder = 'test_result'
        if os.path.isdir(output_folder) is False:
            os.mkdir(output_folder)
        for example in examples:
            example_image = cv2.imread(os.path.join(example_folder_path, example))
            output = os.path.join(output_folder, example.split('.')[0])
            if os.path.isdir(output) is False:
                os.mkdir(output)
            for target in targets:
                print('Process makeup {} based on example model {}'.format(target, example))
                target_image = cv2.imread(os.path.join(target_folder_path,target))
                result = makeup_by_whole_face_transfer.makeup_by_whole_face_transfer(target_image,
                                                                                     example_image,
                                                                                     rectify_landmarks=False)
                cv2.imwrite(os.path.join(output, target.split('.')[0]+'.jpg'), result)
                print('Finish makeup {} based on example model {}'.format(target, example))

if __name__ == '__main__':
    unittest.main()
