import pandas as pd
import numpy as np
import Image
from sklearn import linear_model


def main():
    """
    training.csv has feature points and image data
    test.csv has ImageId and image data

    this task is predicting feature point of test data
    in the format of SampleSubmission.csv (RowId, Location)
    the targets are shown ind
          IdLookupTable.csv (RowId, ImageId, FeatureName, Location)
    """
    #train = pd.read_csv('data/training.csv')
    train = pd.read_csv('data/train_3.csv')
    imgs = []
    # convert image digits

    lec = []
    for i, img_str in enumerate(train.Image):
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
        imgs.append(tmp3)
        # pilImg = Image.fromarray(np.uint8(tmp3))
        # pilImg.show()

    # array[row_s:row_e, col_s:col_e]

    im_lec = []
    sum_matrix = np.zeros((20, 20))
    for i, img in enumerate(imgs):
        p = lec[i]
        roi = img[p[1] - 10:p[1] + 10, p[0] - 10: p[0] + 10]
        sum_matrix += roi
        print roi
        im_lec.append(roi)
        #        print im_lec

    lec_matrix_m = sum_matrix / len(train)
    pilImg = Image.fromarray(np.uint8(lec_matrix_m))
    pilImg.show()
    return


    row_ids = []
    locations = []

    test = pd.read_csv('data/IdLookupTable.csv')
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
