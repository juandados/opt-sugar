from mlflow import pyfunc


def load_model(*args, **kwargs):
    pyfunc_model = pyfunc.load_model(*args, **kwargs)
    pyfunc_model.optimize = pyfunc_model.predict
    return pyfunc_model
