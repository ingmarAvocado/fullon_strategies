class BaseFactory:
    _counter = 0

    @classmethod
    def get_next_id(cls):
        cls._counter += 1
        return cls._counter
