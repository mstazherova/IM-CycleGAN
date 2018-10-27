import tensorflow as tf

class Images():
    def __init__(self, tfrecords, image_size=256, batch_size=1, name=''):
        self.tfrecords = tfrecords
        self.image_size = image_size
        self.batch_size = batch_size
        self.name = name


    def extract_fn(self, data_record):
        features = {'image/encoded_image': tf.FixedLenFeature([], tf.string)}
        sample = tf.parse_single_example(data_record, features)
        image = tf.image.decode_jpeg(sample['image/encoded_image'], channels=3)        
        image = self.preprocess(image)
        
        return image 


    def feed(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(self.extract_fn)
        dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        init = iterator.make_initializer(dataset)
        next_image_data = iterator.get_next()

        return init, next_image_data
    

    def feed_test(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(self.extract_fn)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        init = iterator.make_initializer(dataset)

        return init, iterator
    
     
    def preprocess(self, image):
        image = tf.image.resize_images(image, 
                                       size=(self.image_size, self.image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image / 127.5) - 1.
        image.set_shape([self.image_size, self.image_size, 3])

        return image

    
    def count(self):
        """Counts the number of original images in the TFrecords."""
        num_images = 0 
        record_iterator = tf.python_io.tf_record_iterator(self.tfrecords)
        for example in record_iterator:
            num_images += 1
        
        return num_images

    
    def check_dataset(self, init_op, next_el):
        """Check if images are processed correctly."""
        print("\nChecking dataset ...\n")
        num_images = 0
        with tf.Session() as sess:
            sess.run(init_op)
            try:
                while True:
                    sess.run(next_el)
                    num_images += 1
            except tf.errors.OutOfRangeError:
                pass  
            
        return num_images
