import numpy as np

class Vault:
    def __init__(self, name):
        self.f_name = name
        self.min_right = []
        self.min_left = []
        self.peak_right = []
        self.peak_left = []

    def store(self):
        self.min_values = np.vstack([self.min_right, self.min_left])
        self.peak = np.vstack([self.peak_right, self.peak_left])

        # Write values to csv
        self.matrix_to_csv(self.min_values, f"{self.f_name}_min_values")
        self.matrix_to_csv(self.peak, f"{self.f_name}_peaks")

    def matrix_to_csv(self, matrix, filename):
        np.savetxt(f"csv/{filename}.csv", matrix, delimiter=';', fmt='%i')