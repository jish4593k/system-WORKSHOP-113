import numpy as np
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.png")])
        return file_path

    def load_image(self):
        file_path = self.open_file_dialog()
        if file_path:
            grayscale_image, raw_image = self.get_image(file_path)
            self.display_image(raw_image)
            self.linear_regression(grayscale_image)

    def get_image(self, path):
        image = Image.open(path)
        raw_image = np.copy(image)
        grayscale_image = np.array(image.convert('L'))  # Convert to grayscale
        return grayscale_image, raw_image

    def display_image(self, img):
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(self.root, image=img)
        panel.image = img
        panel.pack()

    def linear_regression(self, data):
        X, y = data[:, 0].reshape(-1, 1), data[:, 1]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Visualize the regression line
        plt.scatter(X, y, color='blue')
        plt.plot(X, y_pred, color='red', linewidth=2)
        plt.title("Linear Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        # Print mean squared error
        mse = mean_squared_error(y, y_pred)
        print(f"Mean Squared Error: {mse}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Processor")
    processor = ImageProcessor(root)
    root.mainloop()
