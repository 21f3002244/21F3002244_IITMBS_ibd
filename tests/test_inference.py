from src.inference import predict

def test_prediction_shape():
    sample = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5]]
    result = predict(sample)
    assert len(result) == 2

def test_prediction_classes():
    sample = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5]]
    result = predict(sample)
    assert set(result).issubset({0, 1, 2})
