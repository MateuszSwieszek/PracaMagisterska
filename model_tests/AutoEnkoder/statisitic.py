import matplotlib.pyplot as plt
import numpy as np
import glob

fileNames = sorted(glob.glob('*.txt'))

dice_dae = np.zeros(shape=(len(fileNames),1976))
dice_unet = np.zeros(shape=(len(fileNames),1976))

for i, fileName in enumerate(fileNames):
    f = open(fileName)
    lines = f.readlines()
    f.close()

    for j, line in enumerate(lines):
        split_lines = line.split(';')
        dice_dae[i,j] = float(split_lines[0])
        dice_unet[i,j] = float(split_lines[1])

    print('###############################################################')
    print('Fold: ', i)
    print('min_dae: ', np.min(dice_dae[i]))
    print('max_dae: ', np.max(dice_dae[i]))
    print('mean_dae: ', np.mean(dice_dae[i]))
    print('standard deviation dae: ', np.std(dice_dae[i]))

    print('###############################################################')
    print('Fold: ', i)
    print('min_unet: ', np.min(dice_unet[i]))
    print('max_unet: ', np.max(dice_unet[i]))
    print('mean_unet: ', np.mean(dice_unet[i]))
    print('standard deviation unet: ', np.std(dice_unet[i]))

dice_dae_list = []
dice_unet_list = []

for i in range(len(fileNames)):
    dice_dae_list.append(dice_dae[i])
    dice_unet_list.append(dice_unet[i])

plt.figure(1)
plt.ylim([0.7, 1.])
plt.boxplot(dice_dae_list)
plt.xticks([1, 2, 3, 4], [0, 1, 2, 3])
plt.xlabel("Folds")
plt.ylabel("Dice coefficient")

plt.savefig("Dice_coeff_dae.png")
plt.figure(2)
plt.ylim([0.7, 1.])
plt.boxplot(dice_unet_list)
plt.xticks([1, 2, 3, 4], [1, 2, 3, 4])
plt.xlabel("Folds")
plt.ylabel("Dice coefficient")
plt.savefig("Dice_coeff_unet.png")
