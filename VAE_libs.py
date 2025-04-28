import tensorflow
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tensorflow.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss



def VAE_hypersapce(img_size , num_channels, latent_space_dim, Initial_CNN_filters,  CNN_filters, kernel_size_shape,strides_size ):
    x = tensorflow.keras.layers.Input(shape=(num_channels, img_size, num_channels), name="encoder_input")
    encoder_conv_layer1 = tensorflow.keras.layers.Conv2D(filters=Initial_CNN_filters, kernel_size=kernel_size_shape, padding="same", strides=strides_size,                                                 name="encoder_conv_1")(x)
    encoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
    encoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)
    encoder_conv_layer2 = tensorflow.keras.layers.Conv2D(filters=CNN_filters, kernel_size=kernel_size_shape, padding="same", strides=strides_size,
                                                         name="encoder_conv_2")(encoder_activ_layer1)
    encoder_norm_layer2 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
    encoder_activ_layer2 = tensorflow.keras.layers.LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)

    encoder_conv_layer3 = tensorflow.keras.layers.Conv2D(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same", strides=2,
                                                         name="encoder_conv_3")(encoder_activ_layer2)
    encoder_norm_layer3 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
    encoder_activ_layer3 = tensorflow.keras.layers.LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

    encoder_conv_layer4 = tensorflow.keras.layers.Conv2D(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same", strides=strides_size*2,
                                                         name="encoder_conv_4")(encoder_activ_layer3)
    encoder_norm_layer4 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
    encoder_activ_layer4 = tensorflow.keras.layers.LeakyReLU(name="encoder_activ_layer_4")(encoder_norm_layer4)

    encoder_conv_layer5 = tensorflow.keras.layers.Conv2D(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same", strides=strides_size,
                                                         name="encoder_conv_5")(encoder_activ_layer4)
    encoder_norm_layer5 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
    encoder_activ_layer5 = tensorflow.keras.layers.LeakyReLU(name="encoder_activ_layer_5")(encoder_norm_layer5)
    shape_before_flatten = tensorflow.keras.backend.int_shape(encoder_activ_layer5)[1:]
    encoder_flatten = tensorflow.keras.layers.Flatten()(encoder_activ_layer5)
    encoder_mu = tensorflow.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
    encoder_log_variance = tensorflow.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(
        encoder_flatten)
    encoder_output = tensorflow.keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])
    encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")


    ## decoder
    decoder_input = tensorflow.keras.layers.Input(shape=(latent_space_dim), name="decoder_input")
    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=np.prod(shape_before_flatten),
                                                         name="decoder_dense_1")(decoder_input)
    decoder_reshape = tensorflow.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)
    decoder_conv_tran_layer1 = tensorflow.keras.layers.Conv2DTranspose(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same",
                                                                       strides=1, name="decoder_conv_tran_1")(decoder_reshape)
    decoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
    decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)
    decoder_conv_tran_layer2 = tensorflow.keras.layers.Conv2DTranspose(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same",
                                                                       strides=strides_size*2, name="decoder_conv_tran_2")(decoder_activ_layer1)
    decoder_norm_layer2 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
    decoder_activ_layer2 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)
    decoder_conv_tran_layer3 = tensorflow.keras.layers.Conv2DTranspose(filters=CNN_filters*2, kernel_size=kernel_size_shape, padding="same",
                                                                       strides=strides_size*2, name="decoder_conv_tran_3")(decoder_activ_layer2)
    decoder_norm_layer3 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
    decoder_activ_layer3 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)
    decoder_conv_tran_layer4 = tensorflow.keras.layers.Conv2DTranspose(filters=Initial_CNN_filters, kernel_size=kernel_size_shape, padding="same",
                                                                       strides=strides_size, name="decoder_conv_tran_4")(decoder_activ_layer3)
    decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4)
    decoder_output1 = tensorflow.keras.layers.Flatten()(decoder_output)
    decoder_output2 = tensorflow.keras.layers.Dense(500)(decoder_output1)
    decoder_output3 = tensorflow.keras.layers.Reshape(target_shape=(1, 500, 1))(decoder_output2)
    decoder = tensorflow.keras.models.Model(decoder_input, decoder_output3, name="decoder_model")


    ### VAE
    vae_input = tensorflow.keras.layers.Input(shape=(num_channels, img_size, num_channels), name="VAE_input")
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = tensorflow.keras.models.Model(vae_input, vae_decoder_output, name="VAE")

    return encoder, encoder_mu, encoder_log_variance, decoder, vae


def load_data():

    validation_data = np.load("validation_data_new.npy")
    validation_label = np.load("validation_label_new.npy")
    training_data = np.load("training_data_new.npy")
    training_label = np.load("training_label_new.npy")
    testing_data = np.load("testing_data_new.npy")
    testing_label = np.load("testing_label_new.npy")

    training_dataa = np.expand_dims(training_data, axis=-1)
    validation_dataa = np.expand_dims(validation_data, axis=-1)
    testing_dataa = np.expand_dims(testing_data, axis=-1)

    training_dataa = np.expand_dims(training_dataa, axis=1)
    validation_dataa = np.expand_dims(validation_dataa, axis=1)
    testing_dataa = np.expand_dims(testing_dataa, axis=1)

    return training_dataa,training_label, validation_dataa, validation_label, testing_dataa, testing_label

def plot_2d(encoded_data, leak_index):

    import plotly.express as px
    # 2D example
    fig = px.scatter(x=encoded_data[:, 0],
                     y=encoded_data[:, 1])
    fig.show("browser")

    trace1 = px.scatter(
        x=encoded_data[:leak_index, 0],
        y=encoded_data[:leak_index, 1],
        color_discrete_sequence=['green']
    )
    trace2 = px.scatter(
        x=encoded_data[leak_index:, 0],
        y=encoded_data[leak_index:, 1],
        color_discrete_sequence=['red']
    )

    fig4 = go.Figure(data=(trace2.data + trace1.data))
    fig4.show("browser")




def plot_3d(encoded_data, leak_index):
    trace1 = px.scatter_3d(
        x=encoded_data[:leak_index, 0],
        y=encoded_data[:leak_index, 1],
        z=encoded_data[:leak_index, 2], color_discrete_sequence=['green']
    )
    trace2 = px.scatter_3d(
        x=encoded_data[leak_index:, 0],
        y=encoded_data[leak_index:, 1],
        z=encoded_data[leak_index:, 2], color_discrete_sequence=['red']
    )

    fig4 = go.Figure(data=(trace1.data + trace2.data ))
    fig4.show("browser")


def smoothTriangle_one_spec(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed



def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    data_smoothed = np.zeros((data.shape[0], data.shape[1]))
    for sample in range(data_smoothed.shape[0]):
        dataa = data[sample]
        smoothed = []
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
        for i in range(degree, len(dataa) - degree * 2):
            point=dataa[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(dataa):
            smoothed.append(smoothed[-1])
        data_smoothed[sample] = NormalizeData(smoothed)
        if sample % 1000 == 0:
            print("sample", sample)
    return data_smoothed

## You need to load own data
training_dataa, training_label, validation_dataa, validation_label, testing_dataa, testing_label = load_data()

### Runing a text of VAE

hyper_space_results = np.zeros((100, 14))

for hyper_test in range(0, 100,1 ):

    CNN_filters = space_CNN_filters[random.randint(0,2) ]
    kernel_size_shape = space_kernel_size_shape[random.randint(0,3)]
    latent_space_dim = space_latent_space_dim[random.randint(0,0)]
    strides_size = space_strides_size[random.randint(0,2)]
    Initial_CNN_filters = space_Initial_CNN_filters[random.randint(0, 2)]
    batch_size = space_batch_size[random.randint(0,2)]

    encoder, encoder_mu, encoder_log_variance, decoder, vae = VAE_hypersapce(img_size, num_channels, latent_space_dim,
                                                                             Initial_CNN_filters, CNN_filters,
                                                                             kernel_size_shape, strides_size)
    vae.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.00001),
                loss=loss_func(encoder_mu, encoder_log_variance))

    history = vae.fit(np.concatenate([training_dataa, validation_dataa], axis=0),
                      np.concatenate([training_dataa, validation_dataa], axis=0), epochs=50, batch_size=128,
                      shuffle=True, validation_data=(testing_dataa, testing_dataa))

    encoded_data_train = encoder.predict(training_dataa)
    encoded_data_val = encoder.predict(validation_dataa)
    encoded_data_test = encoder.predict(testing_dataa)

    #mean_leak_train = encoded_data_train[4000:].sum(0)/4000
    #mean_back_train = encoded_data_train[0:4000].sum(0)/2000

    #mean_leak_val = encoded_data_val[0:240].sum(0)/240
    #mean_back_val = encoded_data_val[240:].sum(0)/705

    #mean_leak_test = encoded_data_test[0:3].sum(0)/3
    #mean_back_test = encoded_data_test[3:].sum(0)/4

    #dst_leak_leak = distance.euclidean(mean_leak_val, mean_leak_test)
    #dst_back_back = distance.euclidean(mean_back_val, mean_back_test)

    #dst_leak_back= distance.euclidean(mean_leak_val, mean_back_test)
    #dst_back_leak = distance.euclidean(mean_back_val, mean_leak_test)

    test_loss = vae.evaluate(testing_dataa, testing_dataa)
    print("test_loss", test_loss)

    hyper_space_results[hyper_test, 0] = history.history["loss"][-1]
    hyper_space_results[hyper_test, 1] = history.history["val_loss"][-1]
    #hyper_space_results[hyper_test, 2] = dst_leak_leak
    #hyper_space_results[hyper_test, 3] = dst_back_back
    #hyper_space_results[hyper_test, 4] = dst_leak_back
    #hyper_space_results[hyper_test, 5] = dst_back_leak
    hyper_space_results[hyper_test, 6] = CNN_filters
    hyper_space_results[hyper_test, 7] = kernel_size_shape[0]
    hyper_space_results[hyper_test, 8] = kernel_size_shape[1]
    hyper_space_results[hyper_test, 9] = latent_space_dim
    hyper_space_results[hyper_test, 10] = strides_size
    hyper_space_results[hyper_test, 11] = Initial_CNN_filters
    hyper_space_results[hyper_test, 12] = batch_size
    hyper_space_results[hyper_test, 13] = test_loss


