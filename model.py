import numpy as np
import tensorflow as tf

def run(content,
        style,
        num_filters=4096,
        alpha=1e-2,
        max_iterations=500
       ):

    num_samples = min(style.shape[1], content.shape[1])
    num_channels = min(style.shape[0], content.shape[0])

    content = content[:num_channels, :num_samples]
    style = style[:num_channels, :num_samples]

    content_tf = np.ascontiguousarray(content.T[None, None, :, :])
    style_tf = np.ascontiguousarray(style.T[None, None, :, :])

    std = np.sqrt(2) * np.sqrt(2.0 / ((num_channels + num_filters) * 11))
    kernel = np.random.randn(1, 11, num_channels, num_filters) * std

    filter_g = tf.Graph()
    with filter_g.as_default(), filter_g.device('/cpu:0'), tf.Session() as sess:
        # data shape is "[batch, in_height, in_width, in_channels]",
        x = tf.placeholder('float32', [1, 1, num_samples, num_channels], name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        net = tf.nn.relu(conv)

        content_features = net.eval(feed_dict={x: content_tf})
        style_features = net.eval(feed_dict={x: style_tf})

        features = np.reshape(style_features, (-1, num_filters))
        style_gram = np.matmul(features.T, features) / num_samples


    result = None

    gen_g = tf.Graph()
    with gen_g.as_default(), gen_g.device('/cpu:0'):
        x = tf.Variable(np.random.randn(1, 1, num_samples, num_channels).astype(np.float32)*1e-3, name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        net = tf.nn.relu(conv)

        content_loss = alpha * 2 * tf.nn.l2_loss(net - content_features)

        style_loss = 0

        _, height, width, channels = map(lambda i: i.value, net.get_shape())

        size = height * width * channels
        feats = tf.reshape(net, (-1, channels))
        gram = tf.matmul(tf.transpose(feats), feats) / num_samples
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

        # Overall loss
        loss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
              loss, method='L-BFGS-B', options={'maxiter': max_iterations})

        # Optimization
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            print('Started optimization')
            opt.minimize(sess)

            print('Final loss:', loss.eval())
            result = x.eval()

    y = np.zeros_like(content)
    y[:num_channels, :] = np.exp(result[0,0].T) - 1

    return y
