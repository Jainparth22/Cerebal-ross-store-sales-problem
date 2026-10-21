"""
Unit Tests: FastAPI Serving Microservice
========================================
Validates API contract schemas, health check responses, and operational invariants.
"""

import unittest
from starlette.testclient import TestClient
from api.main import app, preload_hub_registry


class TestForecastingAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        preload_hub_registry()
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """GET / must return 200 with service information."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documentation", data)
        self.assertEqual(data["status"], "online")

    def test_health_check(self):
        """GET /health must return healthy status and registered model suite."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("LightGBM", data["model_suite"])

    def test_predict_closed_hub_invariant(self):
        """POST /api/v1/forecast with is_open=0 must return strictly 0.0 volume."""
        payload = {
            "hub_id": 1,
            "target_date": "2015-07-01",
            "is_open": 0,
            "promo_active": 0,
            "regional_holiday": 0,
            "school_closure_flag": 0,
        }
        response = self.client.post("/api/v1/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_order_volume"], 0.0)
        self.assertEqual(data["status"], "TEMPORARILY_CLOSED")

    def test_predict_open_hub_returns_positive_volume(self):
        """POST /api/v1/forecast with is_open=1 must return positive volume."""
        payload = {
            "hub_id": 1,
            "target_date": "2015-07-01",
            "is_open": 1,
            "promo_active": 1,
            "regional_holiday": 0,
            "school_closure_flag": 0,
        }
        response = self.client.post("/api/v1/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["predicted_order_volume"], 0.0)
        self.assertEqual(data["status"], "OPERATIONAL")

    def test_batch_forecast_endpoint(self):
        """POST /api/v1/forecast/batch must process multi-record requests."""
        payload = {
            "requests": [
                {"hub_id": 1, "target_date": "2015-07-01", "is_open": 1, "promo_active": 0},
                {"hub_id": 2, "target_date": "2015-07-01", "is_open": 0, "promo_active": 0},
            ]
        }
        response = self.client.post("/api/v1/forecast/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 2)
        self.assertEqual(data["predictions"][1]["predicted_order_volume"], 0.0)


if __name__ == "__main__":
    unittest.main()
