class Settings(object):
    def __init__(self, x_size, y_size, max_n_people):
        self.x_screen_size = x_size
        self.y_screen_size = y_size
        self.keep_playing = True
        self.coords = []
        self.max_people_counter = max_n_people
        for i in range(max_n_people):
            self.coords.append([self.x_screen_size, self.y_screen_size, 0, 0])
        self.people_counter = 0
