import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    fig = plt.figure(figsize=(13, 7))
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax1.plot([1, 2, 3], [4, 5, 6])
    ax2.stackplot([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    ax3.imshow(Image.open('face_classification/images/12_angry_men.jpg'))
    plt.show()
