class StatisticsInput(object):
    def __init__(self):
        self.actions = []

class StatisticsOutputTimestep(object):
    def __init__(self):
        self.iteration = []
        self.increased = []

class StatisticsOutput(object):
    def __init__(self):
        self.done = []
        self.info = []
        self.timestep = StatisticsOutputTimestep()

class Statistics(object):
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.input = StatisticsInput()
        self.output = StatisticsOutput()