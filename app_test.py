from app import tf
from app import np

from app import stats
from app import collect_stat
from app import composed_sample
from app import trim_env_spec_name
from app import is_environments_name
from app import is_environments_list
from app import random_action_space_sample_choice

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

class TimeStepsTestCase(tf.test.TestCase):
	pass

class ArgumentsMock(object):
	def __init__(self, name='sample-data'):
		self.environments = name

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

if __name__ == '__main__':
	tf.test.main()