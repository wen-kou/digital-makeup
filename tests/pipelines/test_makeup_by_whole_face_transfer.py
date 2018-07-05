import unittest
import cv2
import os

from pipelines import makeup_by_whole_face_transfer


class MyTestCase(unittest.TestCase):
    def test_makeup_by_whole_face_transfer(self):
        example_path = '../assets/examples/example_0.jpeg'
        example = cv2.imread(example_path)

        target_path = '../assets/targets/target.jpeg'
        target = cv2.imread(target_path)


        result = makeup_by_whole_face_transfer.makeup_by_whole_face_transfer(target,
                                                                             example,
                                                                             rectify_landmarks=False)
        cv2.imwrite('test_whole_face_transfer.jpg', result)

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
                target_image = cv2.imread(os.path.join(target_folder_path,target))
                result = makeup_by_whole_face_transfer.makeup_by_whole_face_transfer(target_image,
                                                                                     example_image,
                                                                                     rectify_landmarks=False)
                cv2.imwrite(os.path.join(output, target.split('.')[0]+'.jpg'), result)
                print('Finish makeup {} based on example model {}'.format(target, example))

if __name__ == '__main__':
    unittest.main()
