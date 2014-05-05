import pandas as pd
import numpy as np
import Image
from sklearn import linear_model
import sys


def detect(mean_patch, img):
    min = (0, 0, sys.maxint)  # x, y, diff
    for col in xrange(10, 86):
        for row in xrange(10, 86):
            print col, row
            roi = img[row - 10:row + 10, col - 10: col + 10]
            diff_m = mean_patch - roi
            diff = np.sum(np.power(diff_m, 2))
            if diff < min[2]:
                min = (col, row, diff)
    return min[0], min[1]


def main():
    """
    training.csv has feature points and Image
    test.csv has ImageId and image Image

    this task is predicting feature point of test data
    in the format of SampleSubmission.csv (RowId, Location)
    the targets are shown ind
          IdLookupTable.csv (RowId, ImageId, FeatureName, Location)
    """
#    train = pd.read_csv('data/training.csv')
    train = pd.read_csv('data/train_3.csv')
    train_imgs = []
    # convert image digits

    lec = []
    for i, img_str in enumerate(train.Image):
        print 'preproceccing data:', i
        # eye_center
        left_eye_center = (train.left_eye_center_x[i], train.left_eye_center_y[i])
        lec.append(left_eye_center)
        right_eye_center = (train.right_eye_center_x[i], train.right_eye_center_y[i])


        # eye_inner
        left_eye_inner_corner = (train.left_eye_inner_corner_x[i], train.left_eye_inner_corner_y[i])
        right_eye_inner_corner = (train.right_eye_inner_corner_x[i], train.right_eye_inner_corner_y[i])

        # eye_outer
        left_eye_outer_corner = (train.left_eye_outer_corner_x[i], train.left_eye_outer_corner_y[i])
        right_eye_outer_corner = (train.right_eye_outer_corner_x[i], train.right_eye_outer_corner_y[i])


        # eyebrow_inner
        left_eyebrow_inner_end = (train.left_eyebrow_inner_end_x[i], train.left_eyebrow_inner_end_y[i])
        right_eyebrow_inner_end = (train.right_eyebrow_inner_end_x[i], train.right_eyebrow_inner_end_y[i])


        # eyebrow_outer
        left_eyebrow_outer_end = (train.left_eyebrow_outer_end_x[i], train.left_eyebrow_outer_end_y[i])
        right_eyebrow_outer_end = (train.right_eyebrow_outer_end_x[i], train.right_eyebrow_outer_end_y[i])

        # nose_tip
        nose_tip = (train.nose_tip_x[i], train.nose_tip_y[i])

        # mouth_corner
        mouth_left_corner = (train.mouth_left_corner_x[i], train.mouth_left_corner_y[i])
        mouth_right_corner = (train.mouth_right_corner_x[i], train.mouth_right_corner_y[i])

        # mouth_center
        mouth_center_top_lip = (train.mouth_center_top_lip_x[i], train.mouth_center_top_lip_y[i])
        mouth_center_bottom_lip = (train.mouth_center_bottom_lip_x[i], train.mouth_center_bottom_lip_y[i])

        tmp = [int(e) for e in img_str.split()]
        tmp2 = np.matrix(tmp)
        tmp3 = tmp2.reshape((96, 96))
        train_imgs.append(tmp3)
        # pilImg = Image.fromarray(np.uint8(tmp3))
        # pilImg.show()

    test_dic = {}
    # test = pd.read_csv('data/test.csv')
    test = pd.read_csv('data/test_3.csv')
    for img_str, im_id in zip(test.Image, test.ImageId):
        print im_id
        tmp = [int(e) for e in img_str.split()]
        tmp2 = np.matrix(tmp)
        tmp3 = tmp2.reshape((96, 96))
        test_dic[im_id] = tmp3

    # array[row_s:row_e, col_s:col_e]
    im_lec = []
    s_lec_mat = np.zeros((20, 20))
    for i, img in enumerate(train_imgs):
        print 'left_eye:', i
        try:
            p = lec[i]
            roi = img[p[1] - 10:p[1] + 10, p[0] - 10: p[0] + 10]
            s_lec_mat += roi
            im_lec.append(roi)
        except Exception as e:
            print e
        #        print im_lec

    m_lec_mat = s_lec_mat / len(train)

    img = test_dic[1]
    (x, y) = detect(m_lec_mat, img)
    print x, y
    pilImg = Image.fromarray(np.uint8(test_dic[1]))
    pilImg.show()
    return

    row_ids = []
    locations = []


    for row_id, row_featurename in zip(test.RowId, test.FeatureName):
        row_ids.append(row_id)
        locations.append(np.asscalar(train[row_featurename].mean()))

    dic = {
        'RowId': row_ids,
        'Location': locations
    }
    pd.DataFrame(dic).to_csv('res.csv', cols=['RowId', 'Location'], index=False)

if __name__ == '__main__':
    main()
