import unittest
import tensorflow as tf

# Ganti dengan nama file model yang kamu simpan
MODEL_PATH = 'fashion_mnist_model.keras'

class TestFashionMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and normalize test data
        mnist = tf.keras.datasets.fashion_mnist
        (_, _), (cls.x_test, cls.y_test) = mnist.load_data()
        cls.x_test = cls.x_test / 255.0
        # Load trained model dari file
        cls.model = tf.keras.models.load_model(MODEL_PATH)

    def test_model_evaluate(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(loss, 0.0)

    def test_model_predict_shape(self):
        preds = self.model.predict(self.x_test[:10])
        self.assertEqual(preds.shape, (10, 10))
        # Each prediction should sum to 1 (softmax)
        for row in preds:
            self.assertAlmostEqual(sum(row), 1.0, places=3)
        # All probabilities should be between 0 and 1
        self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())

if __name__ == "__main__":
    unittest.main()
