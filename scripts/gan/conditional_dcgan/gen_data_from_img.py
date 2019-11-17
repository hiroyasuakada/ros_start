# generate data from bag images

from PIL import Image
from pathlib import Path
import os, glob  # manipulate file or directory
import numpy as np


class DataArrangement(object):
    def __init__(self):
        self.path = Path(__file__).parent
        self.current_directories = ['not_traking', 'traking']
        self.X_not_traking = []
        self.Y_not_traking = []
        self.X_traking = []
        self.Y_traking = []

    def load_data(self):
        for current_directory in self.current_directories:
            print(current_directory)  # not traking or traking
            self.path /= '../../video_to_image/{}'.format(current_directory)
            directories = os.listdir(self.path)

            for i, directory in enumerate(directories):
                print('{}, {}'.format(i, directory))
                files = glob.glob(str(self.path.resolve()) + '/{}/*.jpg'.format(directory))

                for j, file in enumerate(files):
                    image = Image.open(file)
                    image = image.convert('RGB')
                    # image = image.resize(50, 50)
                    data = np.asarray(image)
                    print('{} - {}'.format(i, j))

                    if current_directory == 'not_traking':  # section off files by directory name
                        self.X_not_traking.append(data)
                        self.Y_not_traking.append(i)
                    else:
                        self.X_traking.append(data)
                        self.Y_traking.append(i)

        return np.array(self.X_not_traking), np.array(self.Y_not_traking), \
               np.array(self.X_traking), np.array(self.Y_traking)


if __name__ == '__main__':
    DA = DataArrangement()
    X_not_traking, Y_not_traking, X_traking, Y_traking = DA.load_data()
