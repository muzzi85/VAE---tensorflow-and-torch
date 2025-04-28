
import tensorflow
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def get_locations(spatial_sampling=2.45):
    # ventouse is air valve
    # vidange is washout
    locations = {
        'Chamber1': int((154 + 120) / spatial_sampling),  # (position + offset)/spatial_sampling
        'Ventouse1': int((154 + 120 + 202) / spatial_sampling),
        'Vidange1': int((154 + 120 + 202 + 115) / spatial_sampling),

        'Chamber2': int((950 + 120) / spatial_sampling),

        'Chamber3': int((1054 + 120) / spatial_sampling),

        'Chamber4': int((1837 + 120) / spatial_sampling),
        'Vidange4': int((1837 + 120 + 309) / spatial_sampling),

        'Chamber5': int((2448 + 120) / spatial_sampling),

        'Chamber6': int((2521 + 120) / spatial_sampling),
        'Vidange6': int((2521 + 120 + 255) / spatial_sampling),

        'Chamber6a': int((2773 + 120) / spatial_sampling),
        'Ventouse6a': int((2521 + 120 + 255 + 270) / spatial_sampling),
        'Vidange6a': int((2521 + 120 + 255 + 270 + 184) / spatial_sampling),

        'Chamber7': int((3560 + 120) / spatial_sampling),
        'Ventouse7': int((3560 + 120 + 269) / spatial_sampling),
        'Vidange7': int((3560 + 120 + 269 + 60) / spatial_sampling),

        'Chamber8': int((4188 + 120) / spatial_sampling),

        # values for C9-C10 link were adjusted after Igors trip to Morocco in 10/09 - 14/09
        'Chamber9': int((4223 + 120) / spatial_sampling),  # chamber 9 was correct

        # location of detected leak on 11/09/2024; real distance measured by Mark, Georgia and Igor from Airvalve
        'Detected_leak_11_09_2024': int((4223 + 120 + 248 - (4 * spatial_sampling) - 8) / spatial_sampling),
        # previous value: 4223+120+248

        'Ventouse9': int((4223 + 120 + 248 - (4 * spatial_sampling)) / spatial_sampling),
        # previous value: 4223+120+248
        'Vidange9': int((4223 + 120 + 248 + 250) / spatial_sampling),
        'dig1_13_09_2024': int((4861 + 120 - (4 * spatial_sampling) - 50) / spatial_sampling),
        'dig2_13_09_2024': int((4861 + 120 - (4 * spatial_sampling) - 40) / spatial_sampling),

        # aux tapping location before chamber 10; real distance measured by Mark, Georgia and Igor
        '80m_to_Chamber10': int((1993)),
        '70m_to_Chamber10': int((1999)),
        '65m_to_Chamber10': int((2001)),
        '60m_to_Chamber10': int((2003)),
        '50m_to_Chamber10': int((2007)),
        '40m_to_Chamber10': int((2011)),
        '30m_to_Chamber10': int((2014)),
        '10m_to_Chamber10': int((2025)),

        'Chamber10': int((4861 + 120 - (4 * spatial_sampling)) / spatial_sampling),  # previous value: 4861+120

        'Chamber11': int((5332 + 120) / spatial_sampling),

        'Chamber12': int((5799 + 120) / spatial_sampling),
        'Vidange12': int((5799 + 120 + 392) / spatial_sampling),

        'Chamber13': int((6453 + 120) / spatial_sampling),
        'Vidange13': int((6453 + 120 + 52) / spatial_sampling),

        'Chamber14': int((7364 + 120) / spatial_sampling),

        'Chamber15': int((7963 + 120) / spatial_sampling),
        'Vidange15': int((7963 + 120 + 173) / spatial_sampling),
        'Ventouse15': int((7963 + 120 + 173 + 926) / spatial_sampling),

        'Chamber16': int((9100 + 120) / spatial_sampling),
    }

    return locations


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



def AI_prediction_VAE (encoder,knn, fft_DAS_signals_F, channale_interval = 1, time_interval = 5, tau = 50, freq_range = [0, 500]):

    AI_files_data = np.zeros((int((fft_DAS_signals_F.shape[0] / (time_interval))) + 5, fft_DAS_signals_F.shape[1], 1, 500,1))

    channel_time = 5

    for channel_idx in range(0, fft_DAS_signals_F.shape[1] - 5, channale_interval):
        count = 0
        for time_idx in range(50, fft_DAS_signals_F.shape[0], time_interval):
            sample_time = np.abs(fft_DAS_signals_F[time_idx - tau:time_idx + tau, channel_idx:channel_idx + channel_time, freq_range[0]:freq_range[1]]).sum(0).sum(0)
            curved_data_time = NormalizeData(sample_time)
            curved_data_time = smoothTriangle_one_spec(curved_data_time, 15)
            curved_data_time = NormalizeData(curved_data_time)

            curved_data_time = np.expand_dims(curved_data_time, axis=0)
            curved_data_time = np.expand_dims(curved_data_time, axis=-1)

            #preds_1_B = model.predict([background1, curved_data_time])
            #preds_1_A = model.predict([background2, curved_data_time])
            #preds_1_C = model.predict([background3, curved_data_time])

            AI_files_data[count, channel_idx, :, :, :] = curved_data_time

            #AI_files_predict[count, channel_idx, 0] = preds_1_B
            #AI_files_predict[count, channel_idx, 1] = preds_1_A
            #AI_files_predict[count, channel_idx, 2] = preds_1_C
            count += 1
            if channel_idx % 100 == 0:
                print("channel_idx", channel_idx)

    AI_files_data = AI_files_data[0:count, :, :, :, :]
    #preds_1_B = model.predict([background1, AI_files_data[0, 123]])
    print(AI_files_data.shape)
    AI_prediction_final = np.zeros((len(AI_files_data),AI_files_data.shape[1]))

    for i in range(len(AI_files_data)):
        # print(i)
        # print(AI_files_data[i])
        # will be each of 7 of axis=0
        # postprocessed_batch = np.abs(time_slice[:, :, :, :])
        postprocessed_batch = np.squeeze(AI_files_data[i], axis=1)
        postprocessed_batch = np.expand_dims(postprocessed_batch, axis=1)
        print(postprocessed_batch.shape)

        #backgrounds_batch0 = np.tile(background1, (postprocessed_batch.shape[0], 1, 1))
        AI_probabilities0 = encoder.predict([ postprocessed_batch])
        knn_pred = knn.predict(np.nan_to_num(AI_probabilities0))
        AI_prediction_final[i, :] = knn_pred
        # break
    #plt.plot(AI_files_data[0, 1866,0])
    #plt.plot(AI_probabilities0 > 0.8)
    #plt.plot(AI_prediction_final[2, :])

    #print(AI_probabilities0)

    return AI_files_data, AI_prediction_final

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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