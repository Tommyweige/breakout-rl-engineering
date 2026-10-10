import numpy as np
import pytest

from breakout_rl.laya_server import LayaDecisionService, create_laya_server, RGB_BYTES


class MockLaya:
    def predict(self, state, questions, **kwargs):
        assert np.asarray(state['image']).shape == (210, 160, 3)
        return {'answers': {'move': {'type': 'choice', 'choice': 'LEFT',
                                    'probabilities': {'NOOP': 0.1, 'FIRE': 0.2, 'RIGHT': 0.3, 'LEFT': 0.4}}}}


def test_server_accepts_only_complete_rgb_and_returns_model_action():
    service = LayaDecisionService(MockLaya(), {'device': 'cuda:0'})
    with pytest.raises(ValueError, match='RGB'):
        service.predict_rgb(b'bad')
    assert service.decisions == 0
    result = service.predict_rgb(bytes(210 * 160 * 3))
    assert result['actionIndex'] == 3 and result['action'] == 'LEFT'
    assert result['probabilities'] == [0.1, 0.2, 0.3, 0.4]
    assert result['device'] == 'cuda:0'
    assert service.decisions == 1
    assert service.policy.decisions == []


def test_http_bridge_rejects_remote_origins_and_incomplete_frames():
    from http.client import HTTPConnection
    from threading import Thread

    service = LayaDecisionService(MockLaya(), {'device': 'cuda:0'})
    server = create_laya_server(service, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection('127.0.0.1', server.server_address[1])
    try:
        client.request('POST', '/api/laya/predict', body=bytes(RGB_BYTES), headers={
            'Origin': 'https://example.com', 'Content-Type': 'application/octet-stream'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.request('POST', '/api/laya/predict', body=b'bad', headers={
            'Origin': 'http://127.0.0.1:5180', 'Content-Type': 'application/octet-stream'})
        response = client.getresponse()
        assert response.status == 400
        response.read()
        assert service.decisions == 0
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_valid_choices_work_without_optional_probability_telemetry():
    class ChoiceOnlyLaya:
        def predict(self, state, questions, **kwargs):
            return {'answers': {'move': {'type': 'choice', 'choice': 'RIGHT'}}}

    result = LayaDecisionService(ChoiceOnlyLaya(), {'device': 'cuda:0'}).predict_rgb(bytes(RGB_BYTES))
    assert result['actionIndex'] == 2
    assert result['probabilities'] is None


def test_unsupported_optimization_falls_back_and_reports_actual_mode(monkeypatch):
    import breakout_rl.laya_browser_agent as module
    def unavailable(*args):
        raise RuntimeError('unsupported template')
    monkeypatch.setattr(module, 'LayaBrowserAgent', unavailable)
    with pytest.warns(UserWarning, match='using reference'):
        service = LayaDecisionService(MockLaya(), {'device': 'cuda:0'}, optimized=True)
    result = service.predict_rgb(bytes(RGB_BYTES))
    assert result['actionIndex'] == 3
    assert result['inferenceMode'] == 'reference'
    assert result['optimizationFallback'] == 'unsupported template'


def test_startup_parity_failure_keeps_reference_decision(monkeypatch):
    import breakout_rl.laya_browser_agent as module
    class DifferentChoice:
        mode = 'fixed-eager'
        def predict(self, *args, **kwargs):
            return {'answers': {'move': {'type': 'choice', 'choice': 'RIGHT',
                    'probabilities': {'NOOP': .1, 'FIRE': .2, 'RIGHT': .4, 'LEFT': .3}}}}
    monkeypatch.setattr(module, 'LayaBrowserAgent', lambda *args: DifferentChoice())
    with pytest.warns(UserWarning, match='differs from reference'):
        service = LayaDecisionService(MockLaya(), {'device': 'cuda:0'}, optimized=True)
    assert service.decisions == 0 and service.policy.decisions == []
    assert service.predict_rgb(bytes(RGB_BYTES))['actionIndex'] == 3
    assert service.runtime['inferenceMode'] == 'reference'
