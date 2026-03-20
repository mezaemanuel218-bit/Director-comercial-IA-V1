class Memory:

    def __init__(self):

        self.history = []

        self.max_messages = 20


    def add(self, role, content):

        self.history.append({
            "role": role,
            "content": content
        })

        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]


    def get(self):

        return self.history


    def clear(self):

        self.history = []