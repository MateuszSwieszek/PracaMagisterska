import matplotlib.pyplot as plt

print("TEST")
from Dice_coeff_Deep_PCA_2 import Modelstatistics
from Dice_coeff_DeepPCA_3 import Modelstatistics_3
import numpy as np


def run():
    masksDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/masks/'
    masksDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/masks/'
    imagesDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/orgs/'
    imagesDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/images/'
    pointsDIR1 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/Orgs/Points/points/'
    pointsDIR2 = '/net/people/plgmswieszek/GenerowanieObrazkow/Images/AlignedImages/Points/points/'
    PCAfile = 'PCA.dat'
    meanFile = 'mean.dat'

    deep_pca_2_results = []
    deep_pca_3_results = []

    for i in range(4):
        modelfile1 = f"/net/people/plgmswieszek/GenerowanieObrazkow/Callbacks/Checkpoints/Models/DeeppPCA_2-{i:04d}.hdf5"
        modelfile2 = f"/net/people/plgmswieszek/GenerowanieObrazkow/DeepPca_3_model-{i:04d}"
        print(modelfile1)
        print(modelfile2)
        print('MODEL1')

        test1 = Modelstatistics(modelfile1, masksDIR1=masksDIR1, masksDIR2=masksDIR2,
                                imagesDIR1=imagesDIR1, imagesDIR2=imagesDIR2,
                                pointsDIR1=pointsDIR1, pointsDIR2=pointsDIR2,
                                PCAfile=PCAfile, meanFile=meanFile)

        print('MODEL2')

        test2 = Modelstatistics_3(modelfile2, masksDIR1=masksDIR1, masksDIR2=masksDIR2,
                                  imagesDIR1=imagesDIR1, imagesDIR2=imagesDIR2,
                                  pointsDIR1=pointsDIR1, pointsDIR2=pointsDIR2,
                                  PCAfile=PCAfile, meanFile=meanFile)
        deep_pca_2_results.append(test1.diceCoeffList)
        deep_pca_3_results.append(test2.diceCoeffList)
        print('###############################################################')
        print('model: ', modelfile1)
        print('min: ', np.min(test1.diceCoeffList))
        print('max: ', np.max(test1.diceCoeffList))
        print('mean: ', np.mean(test1.diceCoeffList))
        print('standard deviation: ', np.std(test1.diceCoeffList))

        print('###############################################################')
        print('model: ', modelfile2)
        print('min: ', np.min(test2.diceCoeffList))
        print('max: ', np.max(test2.diceCoeffList))
        print('mean: ', np.mean(test2.diceCoeffList))
        print('standard deviation: ', np.std(test2.diceCoeffList))

    plt.figure(1)
    plt.ylim([0.50, 1.])
    plt.boxplot(deep_pca_2_results)
    plt.xticks([1, 2, 3, 4], [0, 1, 2,3])
    plt.xlabel("Folds")
    plt.ylabel("Dice coefficient")

    plt.savefig("Dice_coeff_2.png")
    plt.figure(2)
    plt.ylim([0.50, 1.])
    plt.boxplot(deep_pca_3_results)
    plt.xticks([1, 2, 3, 4], [0, 1, 2,3])
    plt.xlabel("Folds")
    plt.ylabel("Dice coefficient")
    plt.savefig("Dice_coeff_3.png")



if __name__ == '__main__':
    run()
