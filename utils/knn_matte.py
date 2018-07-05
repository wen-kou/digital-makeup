import cv2
import numpy as np
import sklearn.neighbors
import warnings
import scipy


def knn_matte(img, trimap, mylambda=100):
    '''
    This is from the implementation of knn matting https://github.com/MarcoForte/knn-matting.git
    '''
    [m, n, c] = img.shape
    img, trimap = img/255.0, trimap/255.0
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background

    # print('Finding nearest neighbors')
    a, b = np.unravel_index(np.arange(m*n), (m, n))
    feature_vec = np.append(np.transpose(img.reshape(m*n, c)), [ a, b]/np.sqrt(m*m + n*n), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    # print('Computing sparse A')
    row_inds = np.repeat(np.arange(m*n), 10)
    col_inds = knns.reshape(m*n*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(m*n, m*n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:,:, 0]))
    v = np.ravel(foreground[:,:,0])
    c = 2*mylambda*np.transpose(v)
    H = 2*(L + mylambda*D)

    # print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)
    return alpha


def get_trimap(input_img, landmarks_list, kernel=None, skip_cut=True):
    landmarks_list = np.asarray(landmarks_list)
    x_min, y_min = np.min(landmarks_list, axis=0)
    x_max, y_max = np.max(landmarks_list, axis=0)
    height = y_max - y_min
    top_left = tuple([x_min, y_min - height if y_min - height >=0 else 0])
    bottom_right = tuple([x_max, y_max + height if y_max + height < input_img.shape[0] else input_img.shape[0]])
    seg_img = input_img[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]
    trimap = np.zeros(input_img.shape)
    cv2.fillPoly(trimap, [landmarks_list], (255,255,255))

    if kernel is None:
        size = int(0.3 * height)
        kernel = np.ones((size, size))
    foreground = cv2.erode(trimap, kernel, iterations=1)
    background = cv2.dilate(trimap, kernel, iterations=1)
    tmp = foreground - background
    ind = np.where(tmp<0)
    trimap[ind[0], ind[1], ind[2]] = 126
    if skip_cut is True:
        return input_img, \
               trimap, \
               {'top_left': tuple([0,0]), 'bottom_right': tuple([input_img.shape[1], input_img.shape[0]])}
    else:
        trimap = trimap[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]
        return seg_img, trimap, {'top_left': top_left, 'bottom_right': bottom_right}


def get_face_trimap(image, facial_contour_landmarks, kernel=None):
    trimap = 126 * np.ones(image.shape)

    trimap = cv2.fillPoly(trimap, [np.asarray(facial_contour_landmarks)], (255, 255, 255))

    x_min, y_min = np.min(facial_contour_landmarks, axis=0)
    x_max, y_max = np.max(facial_contour_landmarks, axis=0)
    height = y_max - y_min
    if kernel is None:
        size = int(0.3 * height)
        kernel = np.ones((size, size))
    trimap = cv2.erode(trimap, kernel, iterations=1)
    y_min = int(y_min - 1.1 * height/3)
    if y_min<0:
        y_min = 0

    trimap[0:y_min+1, :] = (0, 0, 0)
    trimap[:, 0:int(x_min/2)] = (0, 0, 0)
    y_max = int(y_max + 0.5 * (image.shape[0] - y_max))
    if y_max > image.shape[0] - 1:
        y_max = int(image.shape[0] - 1)
    trimap[y_max:image.shape[0], :] = (0, 0, 0)
    x_max = int(x_max + 0.5*(image.shape[1] - x_max))
    if x_max > image.shape[1]-1:
        x_max = image.shape[1]-1
    trimap[:, x_max:image.shape[1]] = (0,0,0)
    return trimap




