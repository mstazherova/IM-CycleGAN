import tensorflow as tf

class Images():
    def __init__(self, tfrecords, image_size=256, batch_size=1, name=''):
        self.tfrecords = tfrecords
        self.image_size = image_size
        self.batch_size = batch_size
        self.name = name


    # def feed(self):
    #     with tf.name_scope(self.name):
    #         filename_queue = tf.train.string_input_producer([self.tfrecords])
    #         reader = tf.TFRecordReader()

    #         _, serialized = reader.read(filename_queue)
    #         features = tf.parse_single_example(
    #             serialized, features={
    #                 'image/file_name': tf.FixedLenFeature([], tf.string),
    #                 'image/encoded_image': tf.FixedLenFeature([], tf.string)})
            
    #         image_buffer = features['image/encoded_image']
    #         image = tf.image.decode_jpeg(image_buffer, channels=3)
    #         image = self.preprocess(image)
    #         images = tf.train.batch([image], batch_size=self.batch_size) 
    #         # tf.summary.image('_input', images)
    #     return images


    def extract_fn(self, data_record):
        features = {'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string)}
        sample = tf.parse_single_example(data_record, features)
        image = tf.image.decode_jpeg(sample['image/encoded_image'], channels=3)        
        # filename = sample['image/file_name']
        image = self.preprocess(image)
        return image 


    # TODO rename this function
    def feed_test(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(self.extract_fn)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        return next_image_data

     

    def preprocess(self, image):
        image = tf.image.resize_images(image, 
                                       size=(self.image_size, self.image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image / 127.5) - 1.
        image.set_shape([self.image_size, self.image_size, 3])

        return image