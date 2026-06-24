import io
import os
import tempfile
import unittest

import backend.app as prism


VALID_CSV = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
1001,P01,Product 1,2,2026-01-01 10:00:00,100,101,India
1002,P02,Product 2,1,2026-01-10 10:00:00,250,101,India
1003,P01,Product 1,3,2026-01-12 10:00:00,80,102,India
1004,P03,Product 3,5,2026-02-01 10:00:00,50,103,India
1005,P04,Product 4,1,2026-02-05 10:00:00,900,104,India
"""


INVALID_CSV = """CustomerName,OrderValue
Alice,100
"""


class PrismApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_paths = {
            "MODEL_PATH": prism.MODEL_PATH,
            "SCALER_PATH": prism.SCALER_PATH,
            "MAP_PATH": prism.MAP_PATH,
            "DATA_PATH": prism.DATA_PATH,
            "DB_PATH": prism.DB_PATH,
        }
        self.original_objects = {
            "model": prism.model,
            "scaler": prism.scaler,
            "persona_map": prism.persona_map,
        }

        prism.MODEL_PATH = os.path.join(self.temp_dir.name, "model.pkl")
        prism.SCALER_PATH = os.path.join(self.temp_dir.name, "scaler.pkl")
        prism.MAP_PATH = os.path.join(self.temp_dir.name, "persona_map.pkl")
        prism.DATA_PATH = os.path.join(self.temp_dir.name, "rfm_segmented.csv")
        prism.DB_PATH = os.path.join(self.temp_dir.name, "prism.db")
        prism.model = None
        prism.scaler = None
        prism.persona_map = None

        prism.app.config["TESTING"] = True
        self.client = prism.app.test_client()

    def tearDown(self):
        for name, value in self.original_paths.items():
            setattr(prism, name, value)
        for name, value in self.original_objects.items():
            setattr(prism, name, value)
        self.temp_dir.cleanup()

    def upload_csv(self, content, filename="transactions.csv"):
        return self.client.post(
            "/upload",
            data={"file": (io.BytesIO(content.encode("utf-8")), filename)},
            content_type="multipart/form-data",
        )

    def test_upload_valid_csv(self):
        response = self.upload_csv(VALID_CSV)

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["message"], "File processed successfully")
        self.assertEqual(payload["customers"], 4)
        self.assertTrue(os.path.exists(prism.DATA_PATH))

    def test_reject_invalid_csv(self):
        response = self.upload_csv(INVALID_CSV)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required transaction columns", response.get_json()["error"])

    def test_metrics_returns_json(self):
        self.upload_csv(VALID_CSV)
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        payload = response.get_json()
        self.assertEqual(payload["total_customers"], 4)
        self.assertIn("avg_recency", payload)

    def test_predict_returns_persona(self):
        self.upload_csv(VALID_CSV)
        response = self.client.post(
            "/predict",
            json={"recency": 10, "frequency": 2, "monetary": 350},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("persona", response.get_json())

    def test_customer_exists(self):
        self.upload_csv(VALID_CSV)
        # Check an existing customer (from VALID_CSV customer IDs e.g. 101)
        response = self.client.get("/api/customer_exists/101")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["exists"])

        # Check a non-existent customer
        response = self.client.get("/api/customer_exists/999")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["exists"])


if __name__ == "__main__":
    unittest.main()
