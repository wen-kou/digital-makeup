import unittest
import cv2
import numpy as np

from face_recognition import face_landmarks

from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from facial_attributes.facial_segment import face_segmentation


class MyTestCase(unittest.TestCase):
    def test_get_triangle_mesh(self):
        image_path = '../assets/examples/example_8.jpeg'
        image = cv2.imread(image_path)

        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, add_points=True)
        lip = np.arange(48,68).tolist()
        landmarks_coords, triangle_indices, triangles = face_segmentation.get_triangle_landmarks(landmarks)
        tmp_triangle_indices = []
        for triangle_index in triangle_indices:
            if (triangle_index[0] in lip) & \
               (triangle_index[1] in lip) & \
               (triangle_index[2] in lip):
                continue
            tmp_triangle_indices.append(triangle_index)

        up_lip_index = np.arange(48, 55)
        up_lip_index = np.hstack((up_lip_index, np.arange(60, 65))).tolist()
        up_lip_landmarks = np.asarray(landmarks)[up_lip_index]
        _, triangle_indices, _ = face_segmentation.get_triangle_landmarks(up_lip_landmarks)

        bottom_contour = [0]
        bottom_contour.extend(np.arange(6, 12).tolist())
        selected_triangle_index = list()
        for i, triangle_index in enumerate(triangle_indices):
            if (triangle_index[0] in bottom_contour) & \
                    (triangle_index[1] in bottom_contour) & \
                    (triangle_index[2] in bottom_contour):
                continue
            selected_triangle_index.append(i)
        triangle_indices = np.asarray(triangle_indices)[selected_triangle_index]
        up_lip_triangle = []
        for triangle_index in triangle_indices:
            tmp = [up_lip_index[triangle_index[0]], up_lip_index[triangle_index[1]],up_lip_index[triangle_index[2]]]
            up_lip_triangle.append(tmp)

        tmp_triangle_indices.extend(up_lip_triangle)

        bottom_lip_index = [48]
        bottom_lip_index.extend(np.arange(54, 61).tolist())
        bottom_lip_index.extend(np.arange(64, 68).tolist())
        bottom_lip_landmarks = np.asarray(landmarks)[bottom_lip_index]
        _, triangle_indices, _ = face_segmentation.get_triangle_landmarks(bottom_lip_landmarks)

        up_contour = np.arange(7, 12).tolist()
        selected_triangle_index = list()
        for i, triangle_index in enumerate(triangle_indices):
            if (triangle_index[0] in up_contour) & \
                    (triangle_index[1] in up_contour) & \
                    (triangle_index[2] in up_contour):
                continue
            selected_triangle_index.append(i)
        triangle_indices = np.asarray(triangle_indices)[selected_triangle_index]
        bottom_lip_triangle = []
        for triangle_index in triangle_indices:
            tmp = [bottom_lip_index[triangle_index[0]], bottom_lip_index[triangle_index[1]], bottom_lip_index[triangle_index[2]]]
            bottom_lip_triangle.append(tmp)

        tmp_triangle_indices.extend(bottom_lip_triangle)

        # remove eyes
        left_eyes = np.arange(36,42).tolist()
        selected_triangle_index =[]
        for i, tmp_triangle in enumerate(tmp_triangle_indices):
            if (tmp_triangle[0] in left_eyes) & \
                    (tmp_triangle[1] in left_eyes) & \
                    (tmp_triangle[2] in left_eyes):
                continue
            else:
                selected_triangle_index.append(i)
        tmp_triangle_indices = np.asarray(tmp_triangle_indices)[selected_triangle_index]

        right_eyes = np.arange(42, 48).tolist()
        selected_triangle_index = []
        for i, tmp_triangle in enumerate(tmp_triangle_indices):
            if (tmp_triangle[0] in right_eyes) & \
                    (tmp_triangle[1] in right_eyes) & \
                    (tmp_triangle[2] in right_eyes):
                continue
            else:
                selected_triangle_index.append(i)
        tmp_triangle_indices = np.asarray(tmp_triangle_indices)[selected_triangle_index]

        np.savetxt('landmark_triangle_index.txt',tmp_triangle_indices, fmt='%i')
        triangles = face_segmentation.get_triangle_mesh(landmarks, tmp_triangle_indices)

        for triangle in triangles:
            cv2.polylines(image, [np.asarray(triangle)],True, (0,255,255))
        cv2.imwrite('test1.jpg', image)
        pass

    def test_face_seg(self):
        triangle_list_path = '../../resources/landmark_triangle_index.txt'
        triangle_list = np.loadtxt(triangle_list_path, dtype=int)
        image_path = '../assets/targets/target_0.jpg'
        image = cv2.imread(image_path)

        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, add_points=True)
        triangles = face_segmentation.get_triangle_mesh(landmarks, triangle_list)

        for triangle in triangles:
            cv2.polylines(image, [np.asarray(triangle)], True, (0, 0, 255))
        cv2.imwrite('test1.jpg', image)
        pass

    def test_calc_triangle_affine_transformation(self):
        sources = [(2,2), (3,3), (0,1)]
        targets = [(2,2), (3,3), (0,1)]
        affine_h = face_segmentation.calc_triangle_affine_transformation(sources, targets)

        self.assertEquals(np.eye(3).tolist(), affine_h.tolist())

        # TODO: more cases should be added

    def test_get_ref_pixels_by_affine_transformation(self):
        sources = np.array([[2, 3, 3], [3, 0, 1]], dtype=np.float32)
        gt = np.array([[0,1,1], [1,-2,-1]])
        affine_h = 0.5 * np.eye(3)
        affine_h[0,2] = -1
        affine_h[1,2] = -1

        res = face_segmentation.pixel_transfer(affine_h, sources)

        self.assertEquals(gt.tolist(), res.tolist())

    def test_get_facial_segments(self):
        image_path = '../assets/examples/example_8.jpeg'
        image = cv2.imread(image_path)

        new_image = np.zeros(image.shape)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, add_points=True)

        chin_landmarks = landmarks[0:17]
        coords = face_segmentation.find_face_region(chin_landmarks,image.shape )
        new_image[coords[0], coords[1], :] = image[coords[0], coords[1], :]
        cv2.imwrite('test1.jpg', new_image)
        pass

    def test_gmm_color_model(self):
        image_path = '../assets/examples/example_8.jpeg'
        image = cv2.imread(image_path)
        image_size = tuple([image.shape[0], image.shape[1]])
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        facial_region_index = face_segmentation.get_facial_indices(landmarks, image_size)
        gmm_color = face_segmentation.gmm_color_model(image, facial_region_index)
        color = np.mean(image[facial_region_index[0], facial_region_index[1], :],axis=0)/255
        image[facial_region_index[0], facial_region_index[1], :] = (255, 255, 255)
        cv2.imwrite('test_facial_region.jpg', image)
        scores = gmm_color.score(np.asarray([color,
                                  [0,0,0]]))
        self.assertGreater(scores[0], scores[1])

    def test_face_recognition(self):
        image_path = '../assets/examples/after-makeup2.jpeg'
        image = cv2.imread(image_path)
        landmarks = face_landmarks(image)[0]
        uplip = landmarks['top_lip']

        landmarks_coords, triangle_indices, triangles = face_segmentation.get_triangle_landmarks(uplip)
        tmp_triangle_indices = []
        # for triangle_index in triangle_indices:
        #     if (triangle_index[0] in inner_mouse) & \
        #        (triangle_index[1] in inner_mouse) & \
        #        (triangle_index[2] in inner_mouse):
        #         continue
        #     tmp_triangle_indices.append(triangle_index)

        # triangles = face_segmentation.get_triangle_mesh(landmarks, tmp_triangle_indices)

        for triangle in triangles:
            cv2.polylines(image, [np.asarray(triangle)],True, (0,255,255))
        cv2.imwrite('test1.jpg', image)
if __name__ == '__main__':
    unittest.main()
