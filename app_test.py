from app import tf

from app import stats

class StatisticsStructAboutGameTestCase(tf.test.TestCase):
	def testObservationsPropertie(self):
		with self.test_session():
			expectation = 'observations' in stats
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