#1. Object Classes

There are three classes for detecting faces of a image: Image, Face, and Organ.

## 1) Image
Attributes:
- self.img: The original image
- self.shape: The shape of the image
- self.faceList: All Face objects detected in this image, None if found nothing

Methods:
- __detect_all_faces()__:
    - Call the _get_all_face_landmark() method, will return all the landmarks of all faces in the image
    - Initialize the Face objects for each face in the image

- ___get_all_face_landmark()__:
    - Use the dlib function to detect 68-point landmarks.
    - Code is copied from the tutorial...
    
## 2) Face
Attributes:
- self.landmark: The 68-point face landmarks
- self.organs: A list of the Organ objects on the face
    - JAW = 0
    - MOUTH = 1
    - NOSE = 2
    - LEFT_EYE = 3
    - RIGHT_EYE = 4
    - LEFT_BROW = 5
    - RIGHT_BROW = 6
- self.img_shape: The shape of the original image, used for initializing masks

Methods:
- ___add_forehead_organ(img)__: 
    - Add the Organ object of the forehead to the Face object. Copied from the tutorial.
    - Input: 
        - _img_: Original image

    
- __get_facemask()__: 
    - Get the mask of the face (forehead + face - eyes - mouth - eyebrows)
    - Output:
        - _whole_face_mask_: The face mask of the face (row, col, channel).


## 3) Organ
Attributes:
- self.landmark: The points that form this organ
- self.name: The organ's name

# 2. Manager Classes
## 1) organmaskMgr:
Methods:
- __update_one_organ(mask, organ_landmark)__: 
    - Input the landmark of an organ, call _get_organ_whole_mask(landmark, image_shape)_ to find the organ's mask, and then add it to the mask.
    - Input:
        - _mask_: Original mask (like a canvas)
        - _organ_landmark_: the landmark of the organ.
    - Output:
        - The updated mask.

- ___get_organ_location(landmark)__: 
    - Input the landmark of the organ, calculate and return the location (coordinate) of this organ as a Location List: [top, bottom, left, right, shape, size, move]
    - Input: 
        - _landmark_: The landmark of an organ
    - Output: 
        - _location_: The Location List of that organ
        
- ___get_organ_patch_mask(landmark, location, img_shape)__: 
    - Input the landmark and Location List of an organ, as well as the original image shape;
    - draw the patch-mask of the organ and return the patch mask.
    - Input:
        - _landmark_: The landmark of an organ
        - _location_: The Location List of that organ
        - _image_shape_: The shape of the original image.
    - Output:
        - _mask_: The __patch mask__ of that organ.

- __get_organ_whole_mask(landmark, image_shape)__:
    - Call the __get_organ_location(landmark)_ to get the location,
    - Call the __get_organ_patch_mask(landmark, location, img_shape)_ to get the patch mask,
    - Copy-paste the patch mask onto the whole-image-size mask.
    - Input:
        - _landmark_: The landmark of an organ,
        - _image_shape_: The shape of the original image.
    - Output:
        - _mask_: The __whole-image-sized__ 3D mask of that organ. 
        
        
- __get_acne_mask(faces, img)__:
    - For each faces, find the left cheek, right cheek and the forehead of the face (Don't use the face mask minus eyes and nose, because it will include the T-zone and the edges around may be detected as acne, which makes the result looks strange)
    - Call the __get_acne_mask_on_patch(img, in_mask)_, find the acne on the cheeks and forehead
    - Append the acne masks of all faces onto the final mask, return that final mask.
    - Input:
        - _faces_: A list of Face objects that are detected in the image
        - _img_: The original image
    - Output:
        - The acne mask of the whole image
        
- ___get_acne_mask_on_patch(img, in_mask)__:
    - Take in the mask within which we search for the acne.
    - Calculate and enhance the image ("Hard Light"*3) and tri-narize the image (convert into three color: white, gray, black)
    - Find the blob in the triary-color image
    - Circle the blob out on the result mask
    - Return the result mask
    - Input:
        - _img_: The original image
        - _in_mask_: The cheek-and-foreheads mask of the image
    - Output:
        - _mask_: The acne-mask of the image
        
## 2) facemaskMgr:
- __get_img_face_mask(img_object)__:
    - Serve as an interface for the outer function to generate a whole-image-sized face mask
    - Input: 
        - _img_object_: The Image object that has already detected the face
    - Output:
        - _whole_img_mask_: The face mask of all the faces in that image
        
        
© July/2018 LzCai