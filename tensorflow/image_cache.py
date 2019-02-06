import random

class ImageCache:
    def __init__(self, cache_size=30):
        self.cache_size = cache_size
        self.images = []

    def fetch(self, image):
        if self.cache_size == 0:
            return image

        p = random.random()
        if p > 0.5 and len(self.images) > 0:
            # use and replace old image.
            random_id = random.randrange(len(self.images))
            retval = self.images[random_id].copy()
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            else:
                self.images[random_id] = image.copy()
            return retval
        else:
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            return image