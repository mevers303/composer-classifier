from model_final import create_model
from file_handlers.dataset import VectorGetterNHot
from keras.utils import plot_model



if __name__ == "__main__":

    dataset = VectorGetterNHot("midi/classical")
    model = create_model(dataset)

    plot_model(model, to_file="models/final.png", show_shapes=True, show_layer_names=False)
    print("Image saved to models/final.png")
