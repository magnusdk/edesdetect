def disable_tensorflow_gpu_usage():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")