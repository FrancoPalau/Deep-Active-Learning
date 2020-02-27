from  keras.utils import plot_model
from keras.models import load_model
py_model=model = load_model('model_test',compile=False)


plot_model(py_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
