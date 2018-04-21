from app import tf

from app import stats

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

class TimeStepsTestCase(tf.test.TestCase):
	pass

class EnvironmentLoggingTestCase(tf.test.TestCase):
	pass

class IterationLoggingTestCase(tf.test.TestCase):
	pass

class RenderingTestCase(tf.test.TestCase):
	pass

class StepsTestCase(tf.test.TestCase):
	pass

class CollectiblesTestCase(tf.test.TestCase):
	pass

class ProgressTestCase(tf.test.TestCase):
	pass

if __name__ == '__main__':
	tf.test.main()