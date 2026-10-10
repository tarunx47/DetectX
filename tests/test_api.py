import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from api import app, history, history_lock


def setup_function():
    """Clear in-memory session history before each test."""
    with history_lock:
        history.clear()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model" in data


def test_initial_empty_stats_and_transactions():
    client = TestClient(app)

    # Stats
    res_stats = client.get("/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["total"] == 0
    assert stats["fraud"] == 0
    assert stats["legitimate"] == 0
    assert stats["fraud_rate"] == 0.0
    assert stats["daily"] == []

    # Transactions
    res_tx = client.get("/transactions")
    assert res_tx.status_code == 200
    txs = res_tx.json()
    assert isinstance(txs, list)
    assert len(txs) == 0


def test_predict_legitimate_transaction():
    client = TestClient(app)
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 50.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 950.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert data["type"] == "PAYMENT"
    assert "label" in data
    assert "risk_level" in data


def test_predict_fraud_transaction():
    client = TestClient(app)
    # Typical PaySim fraud pattern: transfer draining entire balance to 0
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] is not None
    assert data["probability"] >= 0.5
    assert data["label"] == "Fraudulent"
    assert data["risk_level"] in ("High", "Medium")


def test_transactions_and_stats_update_after_prediction():
    client = TestClient(app)
    payload = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    pred_res = client.post("/predict", json=payload)
    assert pred_res.status_code == 200
    tx_id = pred_res.json()["id"]

    # Check transactions endpoint
    tx_res = client.get("/transactions")
    assert tx_res.status_code == 200
    tx_list = tx_res.json()
    assert len(tx_list) == 1
    assert tx_list[0]["id"] == tx_id

    # Check stats endpoint
    stats_res = client.get("/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total"] == 1
    assert stats["fraud"] in (0, 1)
    assert stats["total_transactions"] == 1
    assert len(stats["daily"]) == 1


def test_predict_validation_errors():
    client = TestClient(app)

    # Invalid transaction type
    res_bad_type = client.post("/predict", json={
        "step": 1,
        "type": "BITCOIN",
        "amount": 100.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    })
    assert res_bad_type.status_code == 422

    # Negative amount
    res_neg_amount = client.post("/predict", json={
        "step": 1,
        "type": "TRANSFER",
        "amount": -50.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    })
    assert res_neg_amount.status_code == 422

    # Invalid step (step < 1)
    res_bad_step = client.post("/predict", json={
        "step": 0,
        "type": "TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    })
    assert res_bad_step.status_code == 422


def test_dataset_stats_endpoint():
    client = TestClient(app)
    response = client.get("/dataset-stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 6362620
    assert data["actual_fraud"] == 8213
    assert data["predicted_fraud"] == 77872
    assert "by_type" in data
    assert "evaluation" in data
    assert "precision" in data["evaluation"]
    assert "recall" in data["evaluation"]
    assert "accuracy" in data["evaluation"]


if __name__ == "__main__":
    tests = [
        test_health_endpoint,
        test_initial_empty_stats_and_transactions,
        test_predict_legitimate_transaction,
        test_predict_fraud_transaction,
        test_transactions_and_stats_update_after_prediction,
        test_predict_validation_errors,
        test_dataset_stats_endpoint,
    ]
    passed = 0
    failed = 0
    for t in tests:
        setup_function()
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
