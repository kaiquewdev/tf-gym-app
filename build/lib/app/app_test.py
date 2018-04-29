from app import tf
from app import np

from app import stats
from app import collect_stat
from app import is_action_type
from app import composed_sample
from app import trim_env_spec_name
from app import is_environments_act
from app import is_environments_gen
from app import is_environments_pull
from app import is_environments_name
from app import is_environments_list
from app import random_action_space_sample_choice

from app import DQNAgent
from app import Sequential

from app import deque

from app import Statistics
from app import StatisticsOutput
from app import StatisticsOutputTimestep
from app import StatisticsInput

class DQNAgentTestCase(tf.test.TestCase):
	def testStateSize(self):
		with self.test_session():
			expectation = DQNAgent(10, 30)
			expected = 10
			self.assertEqual(expectation.state_size, 10)

	def testActionSize(self):
		with self.test_session():
			expectation = DQNAgent(10, 30)
			expected = 30
			self.assertEqual(expectation.action_size, 30)

	def testMemory(self):
		with self.test_session():
			expectation = type(DQNAgent(10, 30).memory) == deque
			expected = True
			self.assertEqual(expectation,expected)

	def testGamma(self):
		with self.test_session():
			expectation = DQNAgent(10, 30).gamma
			expected = 0.95
			self.assertEqual(expectation,expected)

	def testEpsilon(self):
		with self.test_session():
			expectation = DQNAgent(10, 30).epsilon
			expected = 1.0
			self.assertEqual(expectation,expected)

	def testEpsilonMin(self):
		with self.test_session():
			expectation = DQNAgent(10, 30).epsilon_min
			expected = 0.01
			self.assertEqual(expectation,expected)

	def testEpsilonDecay(self):
		with self.test_session():
			expectation = DQNAgent(10, 30).epsilon_decay
			expected = 0.99
			self.assertEqual(expectation,expected)

	def testLearningRate(self):
		with self.test_session():
			expectation = DQNAgent(10, 30).learning_rate
			expected = 0.001
			self.assertEqual(expectation,expected)

	def testModelType(self):
		with self.test_session():
			expectation = type(DQNAgent(10, 30).model)
			expected = Sequential
			self.assertEqual(expectation,expected)

	def testTargetModelType(self):
		with self.test_session():
			expectation = type(DQNAgent(10, 30).target_model)
			expected = Sequential
			self.assertEqual(expectation,expected)

	def testUpdateTargetModel(self):
		with self.test_session():
			expectation = type(DQNAgent(10, 30).update_target_model())
			expected = type(None)
			self.assertEqual(expectation,expected)

	def testRemember(self):
		with self.test_session():
			expectation = type(DQNAgent(10, 30).remember(1, 2, 3, 4, 5))
			expected = type(None)
			self.assertEqual(expectation,expected)


class StatisticsStructAboutGameTestCase(tf.test.TestCase):
	def testObservationsProperty(self):
		with self.test_session():
			expectation = 'observations' in stats
			expected = True
			self.assertEqual(expectation, expected)

	def testRewardsProperty(self):
		with self.test_session():
			expectation = 'rewards' in stats
			expected = True
			self.assertEqual(expectation, expected)

	def testInputActionsProperty(self):
		with self.test_session():
			expectation = (('input' in stats) and 'actions' in stats['input'])
			expected = True
			self.assertEqual(expectation, expected)

	def testOutputDoneProperty(self):
		with self.test_session():
			expectation = (('output' in stats) and 'done' in stats['output'])
			expected = True
			self.assertEqual(expectation, expected)

	def testOutputInfoProperty(self):
		with self.test_session():
			expectation = (('output' in stats) and 'info' in stats['output'])
			expected = True
			self.assertEqual(expectation, expected)

	def testOutputTimestepProperties(self):
		with self.test_session():
			expectation = (('output' in stats) and 'timestep' in stats['output'])
			expected = True
			self.assertEqual(expectation, expected)

	def testOutputTimestepIterationProperty(self):
		with self.test_session():
			expectation = ((('output' in stats) and 'timestep' in stats['output']) and 'iteration' in stats['output']['timestep'])
			expected = True
			self.assertEqual(expectation, expected)

	def testOutputTimestepIncreasedProperty(self):
		with self.test_session():
			expectation = ((('output' in stats) and 'timestep' in stats['output']) and 'increased' in stats['output']['timestep'])
			expected = True
			self.assertEqual(expectation, expected)

	def testInstantiation(self):
		with self.test_session():
			expectation = type(Statistics())
			expected = Statistics
			self.assertEqual(expectation, expected)

	def testInstanceObservation(self):
		with self.test_session():
			expectation = type(Statistics().observations)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceRewards(self):
		with self.test_session():
			expectation = type(Statistics().rewards)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceOutput(self):
		with self.test_session():
			expectation = type(StatisticsOutput())
			expected = StatisticsOutput
			self.assertEqual(expectation, expected)

	def testInstanceOutputDone(self):
		with self.test_session():
			expectation = type(Statistics().output.done)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceOutputInfo(self):
		with self.test_session():
			expectation = type(Statistics().output.info)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceOutputTimestep(self):
		with self.test_session():
			expectation = type(StatisticsOutputTimestep())
			expected = StatisticsOutputTimestep
			self.assertEqual(expectation, expected)

	def testInstanceOutputTimestepIteration(self):
		with self.test_session():
			expectation = type(Statistics().output.timestep.iteration)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceOutputTimestepIncreased(self):
		with self.test_session():
			expectation = type(Statistics().output.timestep.increased)
			expected = list
			self.assertEqual(expectation, expected)

	def testInstanceInput(self):
		with self.test_session():
			expectation = type(StatisticsInput())
			expected = StatisticsInput
			self.assertEqual(expectation, expected)

	def testInstanceInputAction(self):
		with self.test_session():
			expectation = type(StatisticsInput().actions)
			expected = list
			self.assertEqual(expectation, expected)

class TimeStepsTestCase(tf.test.TestCase):
	pass

class ArgumentsMock(object):
	def __init__(self, name='sample-data'):
		self.environments = name
		self.action_type = name

class EnvironmentLoggingTestCase(tf.test.TestCase):
	def testRegistryAllSpecNameTreated(self):
		with self.test_session():
			expectation = trim_env_spec_name('EnvSpec(MsPacman-v0)')
			expected = 'MsPacman-v0'
			self.assertEqual(expectation, expected)

	def testIsEnvironmentsName(self):
		with self.test_session():
			expectation = is_environments_name('sample-data', ArgumentsMock())
			expected = True
			self.assertEqual(expectation, expected)

	def testIsEnvironmentsList(self):
		with self.test_session():
			expectation = is_environments_list(ArgumentsMock('list'))
			expected = True
			self.assertEqual(expectation, expected)

	def testIsEnvironmentsAct(self):
		with self.test_session():
			expectation = is_environments_act(ArgumentsMock('act'))
			expected = True
			self.assertEqual(expectation, expected)

	def testIsEnvironmentsGen(self):
		with self.test_session():
			expectation = is_environments_gen(ArgumentsMock('gen'))
			expected = True
			self.assertEqual(expectation, expected)

	def testIsEnvironmentsPull(self):
		with self.test_session():
			expectation = is_environments_pull(ArgumentsMock('pull'))
			expected = True
			self.assertEqual(expectation, expected)

class IterationLoggingTestCase(tf.test.TestCase):
	pass

class RenderingTestCase(tf.test.TestCase):
	pass

class StepsTestCase(tf.test.TestCase):
	pass

class CollectiblesTestCase(tf.test.TestCase):
	def testStatsCollectInputActions(self):
		with self.test_session():
			expectation = collect_stat(10,['input','actions'],stats)
			expected = [10]
			self.assertEqual(expectation,expected)

class ActionSpaceMock(object):
	def sample(self):
		return np.random.randint(5)

class VirtualEnvironmentMock(object):
	def __init__(self):
		self.action_space = ActionSpaceMock()

class ProgressTestCase(tf.test.TestCase):
	def testComposedSampleDefault(self):
		with self.test_session():
			expectation = len(composed_sample(vm=VirtualEnvironmentMock())) == 2
			expected = True
			self.assertEqual(expectation,expected)

	def testEmptyComposedSampleDefault(self):
		with self.test_session():
			expectation = composed_sample()
			expected = []
			self.assertEqual(expectation,expected)

	def testComposedSampleTypes(self):
		with self.test_session():
			expectation = composed_sample(vm=VirtualEnvironmentMock())
			for expected in expectation:
				match = type(expected)
				expected_match = int
				self.assertEqual(match,expected_match)

	def testRandomActionSpaceSampleChoiceDefault(self):
		with self.test_session():
			expectation = type(random_action_space_sample_choice(s=3,vm=VirtualEnvironmentMock()))
			expected = int
			self.assertEqual(expectation,expected)

	def testRandomActionSpaceSampleChoiceRangeReturnedValue(self):
		with self.test_session():
			expectation = random_action_space_sample_choice(s=4,vm=VirtualEnvironmentMock()) >= 0
			expected = True
			self.assertEqual(expectation,expected)

	def testEmptyRandomActionSpaceSampleChoiceDefault(self):
		with self.test_session():
			expectation = random_action_space_sample_choice()
			expected = -1
			self.assertEqual(expectation,expected)

class ProgressTypeTestCase(tf.test.TestCase):
	def testIsActionTypeDefault(self):
		with self.test_session():
			expectation = is_action_type('sample-data', ArgumentsMock())
			expected = True
			self.assertEqual(expectation,expected)

if __name__ == '__main__':
	tf.test.main()