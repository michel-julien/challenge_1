import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lab_access_api_secure as api


class SecureApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = api.app.test_client()
        cls.valid_bytes = (PROJECT_ROOT / "test_valid.png").read_bytes()
        cls.invalid_bytes = (PROJECT_ROOT / "test_invalid.txt").read_bytes()

    def post_image(self, payload, filename):
        return self.client.post(
            "/api/access",
            data={"image": (io.BytesIO(payload), filename)},
            content_type="multipart/form-data",
        )

    def test_missing_image_is_rejected(self):
        response = self.client.post("/api/access")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "No image provided"})

    def test_http_error_gets_generic_json(self):
        response = self.client.get("/route-that-does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {"error": "Invalid request"})

    def test_valid_image_fails_closed_on_simulated_knife(self):
        response = self.post_image(self.valid_bytes, "valid.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"access": False})

    def test_non_image_is_rejected_without_traceback(self):
        response = self.post_image(self.invalid_bytes, "not-an-image.txt")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Invalid image file"})
        self.assertNotIn(b"Traceback", response.data)

    def test_oversized_request_is_rejected(self):
        response = self.post_image(
            b"x" * (api.MAX_REQUEST_BYTES + 1), "oversized.bin"
        )
        self.assertEqual(response.status_code, 413)

    def test_preprocessing_has_expected_tensor_contract(self):
        tensor = api.preprocess_image(self.valid_bytes)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype.name, "float32")
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_response_has_no_internal_fields(self):
        response = self.post_image(self.valid_bytes, "valid.png")
        self.assertEqual(set(response.get_json()), {"access"})

    def test_model_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_model = Path(temporary_directory) / "face_recognition_model.pkl"
            tampered = bytearray((PROJECT_ROOT / "models" / temporary_model.name).read_bytes())
            tampered[-1] = (tampered[-1] + 1) % 256
            temporary_model.write_bytes(tampered)
            original_model_dir = api.MODEL_DIR
            api.MODEL_DIR = Path(temporary_directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    api._load_verified_model(
                        "face_recognition_model.pkl", api.MockFaceRecognitionModel
                    )
            finally:
                api.MODEL_DIR = original_model_dir

    def test_model_failure_returns_generic_503(self):
        class FailingModel:
            def predict(self, _):
                raise RuntimeError("simulated local model failure")

        original_face_model = api.face_model
        api.face_model = FailingModel()
        try:
            response = self.post_image(self.valid_bytes, "valid.png")
        finally:
            api.face_model = original_face_model
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "Access check unavailable"})


if __name__ == "__main__":
    unittest.main()
