import pandas as pd
import numpy as np
import Image
from sklearn import linear_model
import sys


def detect(mean_patch, img):
    min = (0, 0, sys.maxint)  # x, y, diff
    for col in xrange(10, 86):
        for row in xrange(10, 86):
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

    lecs = []
    recs = []

    leis = []
    reis = []

    leos = []
    reos = []

    lebis = []
    rebis = []

    lebos = []
    rebos = []

    nts = []

    mls = []
    mrs = []
    mts = []
    mbs = []

    for i, img_str in enumerate(train.Image):
        print 'preproceccing data:', i
        # eye_center
        lec = (train.left_eye_center_x[i], train.left_eye_center_y[i])
        lecs.append(lec)
        rec = (train.right_eye_center_x[i], train.right_eye_center_y[i])
        recs.append(rec)

        # eye_inner
        lei = (train.left_eye_inner_corner_x[i], train.left_eye_inner_corner_y[i])
        leis.append(lei)
        rei = (train.right_eye_inner_corner_x[i], train.right_eye_inner_corner_y[i])
        reis.append(rei)

        # eye_outer
        leo = (train.left_eye_outer_corner_x[i], train.left_eye_outer_corner_y[i])
        leos.append(leo)
        reo = (train.right_eye_outer_corner_x[i], train.right_eye_outer_corner_y[i])
        reos.append(reo)

        # eyebrow_inner
        lebi = (train.left_eyebrow_inner_end_x[i], train.left_eyebrow_inner_end_y[i])
        lebis.append(lebi)
        rebi = (train.right_eyebrow_inner_end_x[i], train.right_eyebrow_inner_end_y[i])
        rebis.append(rebi)

        # eyebrow_outer
        lebo = (train.left_eyebrow_outer_end_x[i], train.left_eyebrow_outer_end_y[i])
        lebos.append(lebo)
        rebo = (train.right_eyebrow_outer_end_x[i], train.right_eyebrow_outer_end_y[i])
        rebos.append(rebo)

        # nose_tip
        nt = (train.nose_tip_x[i], train.nose_tip_y[i])
        nts.append(nt)

        # mouth_corner
        ml = (train.mouth_left_corner_x[i], train.mouth_left_corner_y[i])
        mls.append(ml)
        mr = (train.mouth_right_corner_x[i], train.mouth_right_corner_y[i])
        mrs.append(mr)

        # mouth_center
        mt = (train.mouth_center_top_lip_x[i], train.mouth_center_top_lip_y[i])
        mts.append(mt)
        mb = (train.mouth_center_bottom_lip_x[i], train.mouth_center_bottom_lip_y[i])
        mbs.append(mb)

        tmp = [int(e) for e in img_str.split()]
        tmp2 = np.matrix(tmp)
        tmp3 = tmp2.reshape((96, 96))
        train_imgs.append(tmp3)
        # pilImg = Image.fromarray(np.uint8(tmp3))
        # pilImg.show()

    test_dic = {}
#    test = pd.read_csv('data/test.csv')
    test = pd.read_csv('data/test_3.csv')
    for img_str, im_id in zip(test.Image, test.ImageId):
        print im_id
        tmp = [int(e) for e in img_str.split()]
        tmp2 = np.matrix(tmp)
        tmp3 = tmp2.reshape((96, 96))
        test_dic[im_id] = tmp3

    # array[row_s:row_e, col_s:col_e]

    ims = []
    points_list = [lecs, recs, leis, reis, leos, reos, lebis, rebis, lebos, rebos, nts, mls, mrs, mts, mbs]
#    points_list = [lecs]
    roi_mat_list = []
    for i_p, points in enumerate(points_list):
        sum_mat = np.zeros((20, 20))
        success_img_cnts = 0
        for i, img in enumerate(train_imgs):
            try:
                p = points[i]
                roi = img[p[1] - 10:p[1] + 10, p[0] - 10: p[0] + 10]
                sum_mat += roi
                success_img_cnts += 1
            except Exception as e:
                pass
                #print e

        roi_mat = sum_mat / success_img_cnts
        roi_mat_list.append(roi_mat)
        pilImg = Image.fromarray(np.uint8(roi_mat))
        pilImg.show()


    img = test_dic[1]
    print '--'
    import ImageDraw
    pilImg = Image.fromarray(np.uint8(test_dic[1]))
    draw = ImageDraw.Draw(pilImg)
    for i in xrange(len(points_list)):
        (x, y) = detect(roi_mat_list[i], img)
        l = (x, y)
        r = (x + 1, y + 1)
        draw.rectangle((l, r), outline='#ff0000', fill='#ff0000')
        print x, y

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
