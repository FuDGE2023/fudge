import numpy as np
import pandas as pd


class DiscreteRegressor:
    def __init__(self, data, filler=0.1, further_length_ratio=0.1, need_regressor=True, seed=123456):

        self.need_regressor = need_regressor
        self.seed = seed
        if self.need_regressor:
            df = pd.DataFrame(np.array(data)[:, 5, 1], columns=['freq'])
            df = df.groupby('freq')['freq'].count()
            d = {'values': df.index, 'count': df.values}
            df = pd.DataFrame(d).sort_values(by=['count', 'values']).reset_index(drop=True)

            length = int(df['values'].max() * (1 + further_length_ratio))

            self.probs = np.full(length, filler, dtype=np.float32)
            self.probs[df['values'] - 1] += df['count']
            self.probs /= self.probs.sum()
            self.values = np.arange(1, length + 1)

    def sample(self, n_samples=1):
        """
        prediction generation

        :param n_samples: number of predictions
        :return: numpy.ndarray predicted values
        """

        if self.need_regressor:
            np.random.seed(self.seed)
            result = np.random.choice(self.values, size=n_samples, p=self.probs)
        else:
            result = np.ones(n_samples)
        return result

