import tensorflow as tf

# 假设你的H5模型文件命名为"model.h5"
h5_model_path = './models/cnn.h5'

# 加载H5模型
model = tf.keras.models.load_model(h5_model_path)

# 指定保存路径
save_dir = './models/cnn'

# 将模型保存为SavedModel格式，这会在指定路径创建文件夹，内含PB文件等相关文件
tf.saved_model.save(model, save_dir)
