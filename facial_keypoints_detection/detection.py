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
    train = pd.read_csv('data/training.csv')

    # convert image digits
    for i, img_str in enumerate(train.Image):
        tmp = [int(e) for e in img_str.split()]
        tmp2 = np.matrix(tmp)
        tmp3 = tmp2.reshape((96, 96))
        pilImg = Image.fromarray(np.uint8(tmp3))
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
